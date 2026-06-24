"""Minimal Vulkan compute host for the KawPow miner.

Wraps device selection (RDNA3 features), buffers, compute pipelines (with forced
wave32 subgroup size, specialization constants and push constants) and synchronous
dispatch. The host language is irrelevant to hashrate -- this only feeds the GPU.
"""

import struct

import vulkan as vk

ffi = vk.ffi

VK_API_1_3 = vk.VK_MAKE_VERSION(1, 3, 0)


def _ver(v):
    return "%d.%d.%d" % ((v >> 22) & 0x7F, (v >> 12) & 0x3FF, v & 0xFFF)


class Buffer:
    def __init__(self, dev, handle, mem, size, mapped_ptr=None):
        self._dev = dev
        self.handle = handle
        self.mem = mem
        self.size = size
        self._ptr = mapped_ptr

    def write(self, data, offset=0):
        assert self._ptr is not None, "buffer is not host-visible"
        self._ptr[offset:offset + len(data)] = data

    def read(self, size=None, offset=0):
        assert self._ptr is not None, "buffer is not host-visible"
        size = self.size if size is None else size
        return bytes(self._ptr[offset:offset + size])

    def destroy(self):
        vk.vkDestroyBuffer(self._dev.device, self.handle, None)
        vk.vkFreeMemory(self._dev.device, self.mem, None)


class ComputePipeline:
    def __init__(self, dev, spirv, num_bindings, push_const_size, local_size,
                 required_subgroup_size=32, entry="main"):
        self._dev = dev
        d = dev.device

        bindings = [vk.VkDescriptorSetLayoutBinding(
            binding=i, descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptorCount=1, stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT)
            for i in range(num_bindings)]
        self.dsl = vk.vkCreateDescriptorSetLayout(
            d, vk.VkDescriptorSetLayoutCreateInfo(pBindings=bindings), None)

        pcr = []
        if push_const_size:
            pcr = [vk.VkPushConstantRange(
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT, offset=0,
                size=push_const_size)]
        self.layout = vk.vkCreatePipelineLayout(
            d, vk.VkPipelineLayoutCreateInfo(pSetLayouts=[self.dsl],
                                             pPushConstantRanges=pcr), None)

        self.module = vk.vkCreateShaderModule(
            d, vk.VkShaderModuleCreateInfo(codeSize=len(spirv), pCode=spirv), None)

        # Specialization: constant_id 0 -> local_size_x. pData is void*, so hand the
        # binding a cffi pointer and keep the backing buffer alive.
        self._spec_data = bytearray(struct.pack("<I", local_size))
        spec = vk.VkSpecializationInfo(
            pMapEntries=[vk.VkSpecializationMapEntry(constantID=0, offset=0, size=4)],
            dataSize=len(self._spec_data),
            pData=ffi.cast("void *", ffi.from_buffer(self._spec_data)))

        rss = vk.VkPipelineShaderStageRequiredSubgroupSizeCreateInfo(
            requiredSubgroupSize=required_subgroup_size)
        stage = vk.VkPipelineShaderStageCreateInfo(
            pNext=rss, stage=vk.VK_SHADER_STAGE_COMPUTE_BIT, module=self.module,
            pName=entry, pSpecializationInfo=spec)
        self.pipeline = vk.vkCreateComputePipelines(
            d, vk.VK_NULL_HANDLE, 1,
            [vk.VkComputePipelineCreateInfo(stage=stage, layout=self.layout)], None)[0]

        # One persistent descriptor set.
        self.pool = vk.vkCreateDescriptorPool(
            d, vk.VkDescriptorPoolCreateInfo(
                maxSets=1, pPoolSizes=[vk.VkDescriptorPoolSize(
                    type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    descriptorCount=max(1, num_bindings))]), None)
        self.dset = vk.vkAllocateDescriptorSets(
            d, vk.VkDescriptorSetAllocateInfo(descriptorPool=self.pool,
                                              pSetLayouts=[self.dsl]))[0]

    def bind(self, buffers):
        writes = [vk.VkWriteDescriptorSet(
            dstSet=self.dset, dstBinding=i, descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[vk.VkDescriptorBufferInfo(buffer=b.handle, offset=0, range=b.size)])
            for i, b in enumerate(buffers)]
        vk.vkUpdateDescriptorSets(self._dev.device, len(writes), writes, 0, None)


class VulkanDevice:
    def __init__(self, device_index=None, prefer_discrete=True):
        app = vk.VkApplicationInfo(pApplicationName="rdna3-kawpow",
                                   applicationVersion=0, pEngineName="rdna3-kawpow",
                                   engineVersion=0, apiVersion=VK_API_1_3)
        self.instance = vk.vkCreateInstance(
            vk.VkInstanceCreateInfo(pApplicationInfo=app), None)

        phys = vk.vkEnumeratePhysicalDevices(self.instance)
        if device_index is not None:
            self.physical = phys[device_index]
        else:
            self.physical = None
            for pd in phys:
                p = vk.vkGetPhysicalDeviceProperties(pd)
                if not prefer_discrete or p.deviceType == vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                    self.physical = pd
                    break
            if self.physical is None:
                self.physical = phys[0]

        props = vk.vkGetPhysicalDeviceProperties(self.physical)
        self.name = props.deviceName.decode() if isinstance(props.deviceName, bytes) else props.deviceName
        self.api_version = _ver(props.apiVersion)

        exts = {e.extensionName.decode() if isinstance(e.extensionName, bytes) else e.extensionName
                for e in vk.vkEnumerateDeviceExtensionProperties(self.physical, None)}
        self.has_subgroup_size_control = "VK_EXT_subgroup_size_control" in exts
        self.has_amd_core2 = "VK_AMD_shader_core_properties2" in exts

        self.subgroup_min, self.subgroup_max, self.compute_units = self._query_props(exts)

        qfams = vk.vkGetPhysicalDeviceQueueFamilyProperties(self.physical)
        self.qfi = next(i for i, q in enumerate(qfams)
                        if q.queueFlags & vk.VK_QUEUE_COMPUTE_BIT)

        ssc = vk.VkPhysicalDeviceSubgroupSizeControlFeatures(
            subgroupSizeControl=vk.VK_TRUE, computeFullSubgroups=vk.VK_FALSE)
        qci = vk.VkDeviceQueueCreateInfo(queueFamilyIndex=self.qfi, queueCount=1,
                                         pQueuePriorities=[1.0])
        dev_exts = []
        if self.has_subgroup_size_control:
            dev_exts.append("VK_EXT_subgroup_size_control")
        self.device = vk.vkCreateDevice(
            self.physical, vk.VkDeviceCreateInfo(
                pNext=ssc, pQueueCreateInfos=[qci],
                ppEnabledExtensionNames=dev_exts), None)
        self.queue = vk.vkGetDeviceQueue(self.device, self.qfi, 0)
        self.cmd_pool = vk.vkCreateCommandPool(
            self.device, vk.VkCommandPoolCreateInfo(
                queueFamilyIndex=self.qfi,
                flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT), None)
        self._memprops = vk.vkGetPhysicalDeviceMemoryProperties(self.physical)

    def _query_props(self, exts):
        sg_min = sg_max = 0
        cus = 0
        try:
            ssc_props = vk.VkPhysicalDeviceSubgroupSizeControlProperties()
            chain = ssc_props
            amd = None
            if "VK_AMD_shader_core_properties2" in exts:
                amd = vk.VkPhysicalDeviceShaderCoreProperties2AMD(pNext=ssc_props)
                chain = amd
            elif "VK_AMD_shader_core_properties" in exts:
                amd = vk.VkPhysicalDeviceShaderCorePropertiesAMD(pNext=ssc_props)
                chain = amd
            props2 = vk.VkPhysicalDeviceProperties2(pNext=chain)
            fn = vk.vkGetInstanceProcAddr(self.instance, "vkGetPhysicalDeviceProperties2")
            if fn:
                vk.vkGetPhysicalDeviceProperties2 = fn
            vk.vkGetPhysicalDeviceProperties2(self.physical, props2)
            sg_min, sg_max = ssc_props.minSubgroupSize, ssc_props.maxSubgroupSize
            # AMD core props expose CU/shader-engine info; available_sgpr etc differ
            # between v1/v2, so just probe the common field defensively.
            try:
                cus = getattr(amd, "activeComputeUnitCount", 0) or 0
            except Exception:
                cus = 0
        except Exception:
            pass
        return sg_min, sg_max, cus

    # --- buffers ---
    def _find_mem(self, type_bits, want):
        for i in range(self._memprops.memoryTypeCount):
            if (type_bits & (1 << i)) and \
               (self._memprops.memoryTypes[i].propertyFlags & want) == want:
                return i
        raise RuntimeError("no suitable memory type")

    def make_buffer(self, size, host_visible=True, storage=True, transfer=False):
        usage = 0
        if storage:
            usage |= vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        if transfer:
            usage |= vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT
        buf = vk.vkCreateBuffer(self.device, vk.VkBufferCreateInfo(
            size=size, usage=usage, sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE), None)
        req = vk.vkGetBufferMemoryRequirements(self.device, buf)
        if host_visible:
            want = (vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        else:
            want = vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        mti = self._find_mem(req.memoryTypeBits, want)
        mem = vk.vkAllocateMemory(self.device, vk.VkMemoryAllocateInfo(
            allocationSize=req.size, memoryTypeIndex=mti), None)
        vk.vkBindBufferMemory(self.device, buf, mem, 0)
        ptr = vk.vkMapMemory(self.device, mem, 0, req.size, 0) if host_visible else None
        return Buffer(self, buf, mem, size, ptr)

    # --- transfers ---
    def copy_buffer(self, src, dst, size, src_offset=0, dst_offset=0):
        """Synchronously copy `size` bytes between two buffers (one region).

        Used to stage the device-local DAG to/from a host-visible buffer for disk
        caching. Callers chunk large transfers so no single submit runs long
        enough to matter for the GPU watchdog.
        """
        cmd = vk.vkAllocateCommandBuffers(self.device, vk.VkCommandBufferAllocateInfo(
            commandPool=self.cmd_pool, level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1))[0]
        vk.vkBeginCommandBuffer(cmd, vk.VkCommandBufferBeginInfo(
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
        region = vk.VkBufferCopy(srcOffset=src_offset, dstOffset=dst_offset, size=size)
        vk.vkCmdCopyBuffer(cmd, src.handle, dst.handle, 1, [region])
        vk.vkEndCommandBuffer(cmd)
        fence = vk.vkCreateFence(self.device, vk.VkFenceCreateInfo(), None)
        vk.vkQueueSubmit(self.queue, 1, [vk.VkSubmitInfo(pCommandBuffers=[cmd])], fence)
        vk.vkWaitForFences(self.device, 1, [fence], vk.VK_TRUE, 10 ** 10)
        vk.vkDestroyFence(self.device, fence, None)
        vk.vkFreeCommandBuffers(self.device, self.cmd_pool, 1, [cmd])

    # --- dispatch ---
    def dispatch(self, pipeline, group_count_x, push_constants=b""):
        cmd = vk.vkAllocateCommandBuffers(self.device, vk.VkCommandBufferAllocateInfo(
            commandPool=self.cmd_pool, level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1))[0]
        vk.vkBeginCommandBuffer(cmd, vk.VkCommandBufferBeginInfo(
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline)
        vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                                   pipeline.layout, 0, 1, [pipeline.dset], 0, None)
        if push_constants:
            pc_buf = ffi.from_buffer(bytearray(push_constants))
            vk.vkCmdPushConstants(cmd, pipeline.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT,
                                  0, len(push_constants), ffi.cast("void *", pc_buf))
        vk.vkCmdDispatch(cmd, group_count_x, 1, 1)
        vk.vkEndCommandBuffer(cmd)
        fence = vk.vkCreateFence(self.device, vk.VkFenceCreateInfo(), None)
        vk.vkQueueSubmit(self.queue, 1, [vk.VkSubmitInfo(pCommandBuffers=[cmd])], fence)
        vk.vkWaitForFences(self.device, 1, [fence], vk.VK_TRUE, 10 ** 10)
        vk.vkDestroyFence(self.device, fence, None)
        vk.vkFreeCommandBuffers(self.device, self.cmd_pool, 1, [cmd])

    def summary(self):
        return (f"{self.name}  (Vulkan {self.api_version})\n"
                f"  subgroup size control: {self.has_subgroup_size_control} "
                f"[{self.subgroup_min}..{self.subgroup_max}]\n"
                f"  AMD core props2: {self.has_amd_core2}  "
                f"compute units: {self.compute_units or '?'}")
