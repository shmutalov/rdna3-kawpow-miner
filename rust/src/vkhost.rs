//! Minimal Vulkan compute host (ash port of `rdna3_kawpow/vkhost.py`).
//!
//! Wraps device selection (RDNA3 feature probe), host-visible/device-local
//! buffers, compute pipelines (forced wave32 via `requiredSubgroupSize=32`, a
//! specialization-constant workgroup size, push constants) and synchronous
//! dispatch. The host language is irrelevant to hashrate -- this only feeds the GPU.
//!
//! Ownership: a single `Arc<DeviceCtx>` holds the instance/device/queue/command
//! pool; `Buffer` and `ComputePipeline` each keep a clone, so the device is torn
//! down only after every resource that uses it has been dropped.

use std::ffi::CStr;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use ash::{vk, Entry, Instance};

const VK_API: u32 = vk::API_VERSION_1_3;

fn ver_str(v: u32) -> String {
    format!(
        "{}.{}.{}",
        vk::api_version_major(v),
        vk::api_version_minor(v),
        vk::api_version_patch(v)
    )
}

/// One physical device, for listing / counting (no logical device created).
pub struct DeviceInfo {
    pub index: usize,
    pub name: String,
    pub discrete: bool,
    /// PCI bus number (VK_EXT_pci_bus_info), for mapping to HiveOS GPU slots.
    pub pci_bus: Option<u32>,
}

/// True if `physical` advertises `needle` as a device extension.
fn device_has_ext(instance: &Instance, physical: vk::PhysicalDevice, needle: &CStr) -> bool {
    unsafe { instance.enumerate_device_extension_properties(physical) }
        .map(|exts| {
            exts.iter()
                .any(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) } == needle)
        })
        .unwrap_or(false)
}

/// Query the GPU's PCI bus number via VK_EXT_pci_bus_info (None if unsupported).
fn query_pci_bus(instance: &Instance, physical: vk::PhysicalDevice) -> Option<u32> {
    if !device_has_ext(instance, physical, ash::ext::pci_bus_info::NAME) {
        return None;
    }
    let mut pci = vk::PhysicalDevicePCIBusInfoPropertiesEXT::default();
    let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut pci);
    unsafe { instance.get_physical_device_properties2(physical, &mut props2) };
    Some(pci.pci_bus)
}

/// Enumerate all Vulkan physical devices (for multi-GPU rig orchestration).
pub fn enumerate_devices() -> Result<Vec<DeviceInfo>> {
    let entry = unsafe { Entry::load() }
        .context("failed to load the Vulkan loader (is a Vulkan ICD installed?)")?;
    let app_name = c"rdna3-kawpow";
    let app_info = vk::ApplicationInfo::default()
        .application_name(app_name)
        .engine_name(app_name)
        .api_version(VK_API);
    let instance = unsafe {
        entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app_info), None)
    }
    .context("vkCreateInstance failed")?;

    let result = (|| -> Result<Vec<DeviceInfo>> {
        let phys = unsafe { instance.enumerate_physical_devices() }?;
        Ok(phys
            .iter()
            .enumerate()
            .map(|(index, &pd)| {
                let props = unsafe { instance.get_physical_device_properties(pd) };
                let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
                    .to_string_lossy()
                    .into_owned();
                DeviceInfo {
                    index,
                    name,
                    discrete: props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU,
                    pci_bus: query_pci_bus(&instance, pd),
                }
            })
            .collect())
    })();
    unsafe { instance.destroy_instance(None) };
    result
}

/// Shared device context. Dropped last (after all `Buffer`/`ComputePipeline`).
struct DeviceCtx {
    device: ash::Device,
    instance: Instance,
    physical: vk::PhysicalDevice,
    queue: vk::Queue,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    cmd_pool: vk::CommandPool,
    _entry: Entry,
}

impl Drop for DeviceCtx {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_command_pool(self.cmd_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
        let _ = self.physical;
    }
}

/// A GPU buffer (optionally host-visible+coherent and persistently mapped).
pub struct Buffer {
    ctx: Arc<DeviceCtx>,
    pub handle: vk::Buffer,
    mem: vk::DeviceMemory,
    pub size: u64,
    mapped: *mut u8, // null if device-local
}

impl Buffer {
    pub fn write(&self, data: &[u8], offset: usize) {
        assert!(!self.mapped.is_null(), "buffer is not host-visible");
        assert!(offset + data.len() <= self.size as usize, "buffer write OOB");
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.mapped.add(offset), data.len());
        }
    }

    pub fn read_into(&self, dst: &mut [u8], offset: usize) {
        assert!(!self.mapped.is_null(), "buffer is not host-visible");
        assert!(offset + dst.len() <= self.size as usize, "buffer read OOB");
        unsafe {
            std::ptr::copy_nonoverlapping(self.mapped.add(offset), dst.as_mut_ptr(), dst.len());
        }
    }

    pub fn read(&self, size: usize, offset: usize) -> Vec<u8> {
        let mut v = vec![0u8; size];
        self.read_into(&mut v, offset);
        v
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.ctx.device.destroy_buffer(self.handle, None);
            self.ctx.device.free_memory(self.mem, None);
        }
    }
}

/// A compute pipeline with one persistent descriptor set (N storage buffers).
pub struct ComputePipeline {
    ctx: Arc<DeviceCtx>,
    dsl: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    module: vk::ShaderModule,
    pipeline: vk::Pipeline,
    pool: vk::DescriptorPool,
    dset: vk::DescriptorSet,
}

impl ComputePipeline {
    pub fn new(
        dev: &VulkanDevice,
        spirv: &[u8],
        num_bindings: u32,
        push_const_size: u32,
        local_size: u32,
        required_subgroup_size: u32,
    ) -> Result<Self> {
        let ctx = dev.ctx.clone();
        let d = &ctx.device;

        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..num_bindings)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let dsl = unsafe {
            d.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }?;

        let pcr = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(push_const_size)];
        let set_layouts = [dsl];
        let mut layout_info =
            vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        if push_const_size > 0 {
            layout_info = layout_info.push_constant_ranges(&pcr);
        }
        let layout = unsafe { d.create_pipeline_layout(&layout_info, None) }?;

        let code = ash::util::read_spv(&mut std::io::Cursor::new(spirv))
            .context("SPIR-V is not a valid word stream")?;
        let module = unsafe {
            d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)
        }?;

        // Specialization: constant_id 0 -> local_size_x.
        let spec_data = local_size.to_ne_bytes();
        let spec_entries = [vk::SpecializationMapEntry::default()
            .constant_id(0)
            .offset(0)
            .size(4)];
        let spec_info = vk::SpecializationInfo::default()
            .map_entries(&spec_entries)
            .data(&spec_data);

        let mut rss = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
            .required_subgroup_size(required_subgroup_size);
        let entry = c"main";
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(entry)
            .specialization_info(&spec_info)
            .push_next(&mut rss);

        let pipe_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout);
        let pipeline = unsafe {
            d.create_compute_pipelines(vk::PipelineCache::null(), &[pipe_info], None)
        }
        .map_err(|(_, e)| anyhow!("vkCreateComputePipelines failed: {e:?}"))?[0];

        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(num_bindings.max(1))];
        let pool = unsafe {
            d.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&pool_sizes),
                None,
            )
        }?;
        let dset = unsafe {
            d.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&set_layouts),
            )
        }?[0];

        Ok(ComputePipeline {
            ctx,
            dsl,
            layout,
            module,
            pipeline,
            pool,
            dset,
        })
    }

    /// Point the descriptor set at `buffers` (binding i -> buffers[i]).
    pub fn bind(&self, buffers: &[&Buffer]) {
        let infos: Vec<vk::DescriptorBufferInfo> = buffers
            .iter()
            .map(|b| {
                vk::DescriptorBufferInfo::default()
                    .buffer(b.handle)
                    .offset(0)
                    .range(b.size)
            })
            .collect();
        let writes: Vec<vk::WriteDescriptorSet> = infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(self.dset)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect();
        unsafe { self.ctx.device.update_descriptor_sets(&writes, &[]) };
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        unsafe {
            let d = &self.ctx.device;
            d.destroy_descriptor_pool(self.pool, None);
            d.destroy_pipeline(self.pipeline, None);
            d.destroy_shader_module(self.module, None);
            d.destroy_pipeline_layout(self.layout, None);
            d.destroy_descriptor_set_layout(self.dsl, None);
        }
    }
}

/// Intermediate produced by `probe_and_create`.
struct DeviceParts {
    name: String,
    api_version: String,
    has_subgroup_size_control: bool,
    has_amd_core2: bool,
    subgroup_min: u32,
    subgroup_max: u32,
    compute_units: u32,
    pci_bus: Option<u32>,
    queue: vk::Queue,
    device: ash::Device,
    physical: vk::PhysicalDevice,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    cmd_pool: vk::CommandPool,
}

/// A selected RDNA3 GPU with a created logical device + compute queue.
pub struct VulkanDevice {
    ctx: Arc<DeviceCtx>,
    pub name: String,
    pub api_version: String,
    pub has_subgroup_size_control: bool,
    pub has_amd_core2: bool,
    pub subgroup_min: u32,
    pub subgroup_max: u32,
    pub compute_units: u32,
    /// PCI bus number (VK_EXT_pci_bus_info), for HiveOS GPU-slot mapping.
    pub pci_bus: Option<u32>,
}

impl VulkanDevice {
    pub fn new(device_index: Option<usize>) -> Result<Self> {
        Self::new_with(device_index, true)
    }

    pub fn new_with(device_index: Option<usize>, prefer_discrete: bool) -> Result<Self> {
        let entry = unsafe { Entry::load() }
            .context("failed to load the Vulkan loader (is a Vulkan ICD installed?)")?;

        let app_name = c"rdna3-kawpow";
        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name)
            .engine_name(app_name)
            .api_version(VK_API);
        let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance = unsafe { entry.create_instance(&create_info, None) }
            .context("vkCreateInstance failed")?;

        let parts = match Self::probe_and_create(&instance, device_index, prefer_discrete) {
            Ok(p) => p,
            Err(e) => {
                unsafe { instance.destroy_instance(None) };
                return Err(e);
            }
        };

        let ctx = Arc::new(DeviceCtx {
            device: parts.device,
            instance,
            physical: parts.physical,
            queue: parts.queue,
            mem_props: parts.mem_props,
            cmd_pool: parts.cmd_pool,
            _entry: entry,
        });

        Ok(VulkanDevice {
            ctx,
            name: parts.name,
            api_version: parts.api_version,
            has_subgroup_size_control: parts.has_subgroup_size_control,
            has_amd_core2: parts.has_amd_core2,
            subgroup_min: parts.subgroup_min,
            subgroup_max: parts.subgroup_max,
            compute_units: parts.compute_units,
            pci_bus: parts.pci_bus,
        })
    }

    fn probe_and_create(
        instance: &Instance,
        device_index: Option<usize>,
        prefer_discrete: bool,
    ) -> Result<DeviceParts> {
        let phys = unsafe { instance.enumerate_physical_devices() }?;
        if phys.is_empty() {
            return Err(anyhow!("no Vulkan physical devices found"));
        }
        let physical = match device_index {
            Some(i) => *phys
                .get(i)
                .ok_or_else(|| anyhow!("device index {i} out of range (have {})", phys.len()))?,
            None => phys
                .iter()
                .copied()
                .find(|&pd| {
                    let p = unsafe { instance.get_physical_device_properties(pd) };
                    !prefer_discrete || p.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
                })
                .unwrap_or(phys[0]),
        };

        let props = unsafe { instance.get_physical_device_properties(physical) };
        let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let api_version = ver_str(props.api_version);

        let ext_props = unsafe { instance.enumerate_device_extension_properties(physical) }?;
        let has_ext = |needle: &CStr| -> bool {
            ext_props
                .iter()
                .any(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) } == needle)
        };
        let has_subgroup_size_control = has_ext(ash::ext::subgroup_size_control::NAME);
        let has_amd_core2 = has_ext(ash::amd::shader_core_properties2::NAME);
        let has_pci_bus = has_ext(ash::ext::pci_bus_info::NAME);
        let pci_bus = query_pci_bus(instance, physical);

        let mut ssc_props = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
        let mut amd_core2 = vk::PhysicalDeviceShaderCoreProperties2AMD::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut ssc_props);
        if has_amd_core2 {
            props2 = props2.push_next(&mut amd_core2);
        }
        unsafe { instance.get_physical_device_properties2(physical, &mut props2) };
        let subgroup_min = ssc_props.min_subgroup_size;
        let subgroup_max = ssc_props.max_subgroup_size;
        let compute_units = if has_amd_core2 {
            amd_core2.active_compute_unit_count
        } else {
            0
        };

        let qfams = unsafe { instance.get_physical_device_queue_family_properties(physical) };
        let qfi = qfams
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .ok_or_else(|| anyhow!("no compute queue family"))? as u32;

        let priorities = [1.0_f32];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(qfi)
            .queue_priorities(&priorities);
        let mut ssc_features =
            vk::PhysicalDeviceSubgroupSizeControlFeatures::default().subgroup_size_control(true);

        let mut dev_exts: Vec<*const std::os::raw::c_char> = Vec::new();
        if has_subgroup_size_control {
            dev_exts.push(ash::ext::subgroup_size_control::NAME.as_ptr());
        }
        if has_pci_bus {
            dev_exts.push(ash::ext::pci_bus_info::NAME.as_ptr());
        }
        let dci = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_extension_names(&dev_exts)
            .push_next(&mut ssc_features);
        let device = unsafe { instance.create_device(physical, &dci, None) }
            .context("vkCreateDevice failed")?;
        let queue = unsafe { device.get_device_queue(qfi, 0) };
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical) };
        let cmd_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(qfi)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }?;

        Ok(DeviceParts {
            name,
            api_version,
            has_subgroup_size_control,
            has_amd_core2,
            subgroup_min,
            subgroup_max,
            compute_units,
            pci_bus,
            queue,
            device,
            physical,
            mem_props,
            cmd_pool,
        })
    }

    fn find_mem(&self, type_bits: u32, want: vk::MemoryPropertyFlags) -> Result<u32> {
        let mp = &self.ctx.mem_props;
        for i in 0..mp.memory_type_count {
            if (type_bits & (1 << i)) != 0
                && mp.memory_types[i as usize].property_flags.contains(want)
            {
                return Ok(i);
            }
        }
        Err(anyhow!("no suitable memory type"))
    }

    pub fn make_buffer(
        &self,
        size: u64,
        host_visible: bool,
        storage: bool,
        transfer: bool,
    ) -> Result<Buffer> {
        let mut usage = vk::BufferUsageFlags::empty();
        if storage {
            usage |= vk::BufferUsageFlags::STORAGE_BUFFER;
        }
        if transfer {
            usage |= vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        }
        let d = &self.ctx.device;
        let handle = unsafe {
            d.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
        }?;
        let req = unsafe { d.get_buffer_memory_requirements(handle) };
        let want = if host_visible {
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        } else {
            vk::MemoryPropertyFlags::DEVICE_LOCAL
        };
        let mti = match self.find_mem(req.memory_type_bits, want) {
            Ok(i) => i,
            Err(e) => {
                unsafe { d.destroy_buffer(handle, None) };
                return Err(e);
            }
        };
        let mem = unsafe {
            d.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(mti),
                None,
            )
        }?;
        unsafe { d.bind_buffer_memory(handle, mem, 0) }?;
        let mapped = if host_visible {
            unsafe { d.map_memory(mem, 0, req.size, vk::MemoryMapFlags::empty()) }? as *mut u8
        } else {
            std::ptr::null_mut()
        };
        Ok(Buffer {
            ctx: self.ctx.clone(),
            handle,
            mem,
            size,
            mapped,
        })
    }

    /// One-time submit of a recorded command buffer, blocking on a fence.
    fn submit_blocking(&self, record: impl FnOnce(vk::CommandBuffer)) -> Result<()> {
        let d = &self.ctx.device;
        let cmd = unsafe {
            d.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.ctx.cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }?[0];
        unsafe {
            d.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }
        record(cmd);
        unsafe { d.end_command_buffer(cmd)? };
        let fence = unsafe { d.create_fence(&vk::FenceCreateInfo::default(), None) }?;
        let cmds = [cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&cmds);
        let res = (|| -> Result<()> {
            unsafe {
                d.queue_submit(self.ctx.queue, &[submit], fence)?;
                d.wait_for_fences(&[fence], true, u64::MAX)?;
            }
            Ok(())
        })();
        unsafe {
            d.destroy_fence(fence, None);
            d.free_command_buffers(self.ctx.cmd_pool, &[cmd]);
        }
        res
    }

    /// Synchronously copy `size` bytes between two buffers (one region).
    pub fn copy_buffer(
        &self,
        src: &Buffer,
        dst: &Buffer,
        size: u64,
        src_offset: u64,
        dst_offset: u64,
    ) -> Result<()> {
        let region = vk::BufferCopy::default()
            .src_offset(src_offset)
            .dst_offset(dst_offset)
            .size(size);
        self.submit_blocking(|cmd| unsafe {
            self.ctx
                .device
                .cmd_copy_buffer(cmd, src.handle, dst.handle, &[region]);
        })
    }

    /// Synchronously dispatch `group_count_x` workgroups of `pipeline`.
    pub fn dispatch(
        &self,
        pipeline: &ComputePipeline,
        group_count_x: u32,
        push_constants: &[u8],
    ) -> Result<()> {
        let d = &self.ctx.device;
        self.submit_blocking(|cmd| unsafe {
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
            d.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.layout,
                0,
                &[pipeline.dset],
                &[],
            );
            if !push_constants.is_empty() {
                d.cmd_push_constants(
                    cmd,
                    pipeline.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push_constants,
                );
            }
            d.cmd_dispatch(cmd, group_count_x, 1, 1);
        })
    }

    pub fn summary(&self) -> String {
        let cus = if self.compute_units != 0 {
            self.compute_units.to_string()
        } else {
            "?".to_string()
        };
        let bus = self.pci_bus.map(|b| b.to_string()).unwrap_or_else(|| "?".into());
        format!(
            "{}  (Vulkan {})\n  \
             subgroup size control: {} [{}..{}]\n  \
             AMD core props2: {}  compute units: {}  PCI bus: {}",
            self.name,
            self.api_version,
            self.has_subgroup_size_control,
            self.subgroup_min,
            self.subgroup_max,
            self.has_amd_core2,
            cus,
            bus,
        )
    }
}
