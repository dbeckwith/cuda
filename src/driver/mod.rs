//! CUDA driver
//!
//! Reference: http://docs.nvidia.com/cuda/cuda-driver-api/

use std::ffi::CStr;
use std::marker::PhantomData;
use std::{mem, ptr, result};

#[allow(dead_code)]
#[allow(non_camel_case_types)]
#[allow(non_snake_case)]
#[allow(non_upper_case_globals)]
mod ll;

/// A CUDA "block"
pub struct Block {
    /// The width of the block in threads
    pub x: u32,
    /// The height of the block in threads
    pub y: u32,
    /// The depth of the block in threads
    pub z: u32,
}

impl Block {
    /// One dimensional block
    pub fn x(x: u32) -> Self {
        Block { x: x, y: 1, z: 1 }
    }

    /// Two dimensional block
    pub fn xy(x: u32, y: u32) -> Self {
        Block { x: x, y: y, z: 1 }
    }

    /// Three dimensional block
    pub fn xyz(x: u32, y: u32, z: u32) -> Self {
        Block { x: x, y: y, z: z }
    }
}

/// A CUDA "context"
#[derive(Debug)]
pub struct Context {
    defused: bool,
    handle: ll::CUcontext,
}

impl Context {
    // TODO is this actually useful? Note that we are using "RAII" (cf. `drop`)
    // and ownership to manage `Context`es
    #[allow(dead_code)]
    fn current() -> Result<Option<Self>> {
        let mut handle = ptr::null_mut();

        unsafe { lift(ll::cuCtxGetCurrent(&mut handle))? }

        if handle.is_null() {
            Ok(None)
        } else {
            Ok(Some(Context {
                defused: true,
                handle: handle,
            }))
        }
    }

    /// Binds context to the calling thread
    pub fn set_current(&self) -> Result<()> {
        unsafe {
            lift(ll::cuCtxSetCurrent(self.handle))
        }
    }

    /// Loads a PTX module
    pub fn load_module<'ctx>(&'ctx self, image: &CStr) -> Result<Module<'ctx>> {
        let mut handle = ptr::null_mut();

        unsafe {
            lift(ll::cuModuleLoadData(&mut handle, image.as_ptr() as *const _))?
        }

        Ok(Module {
            handle: handle,
            _context: PhantomData,
        })
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if !self.defused {
            let _ignored = unsafe { lift(ll::cuCtxDestroy_v2(self.handle)) };
        }
    }
}

/// A CUDA device (a GPU)
#[derive(Debug)]
pub struct Device {
    handle: ll::CUdevice,
}

/// Binds to the `nth` device
#[allow(non_snake_case)]
pub fn Device(nth: u16) -> Result<Device> {
    let mut handle = 0;

    unsafe { lift(ll::cuDeviceGet(&mut handle, i32::from(nth)))? }

    Ok(Device { handle: handle })
}

impl Device {
    /// Returns the number of available devices
    pub fn count() -> Result<u32> {
        let mut count: i32 = 0;

        unsafe { lift(ll::cuDeviceGetCount(&mut count))? }

        Ok(count as u32)
    }

    /// Creates a CUDA context for this device
    pub fn create_context(&self) -> Result<Context> {
        let mut handle = ptr::null_mut();
        // TODO expose
        let flags = 0;

        unsafe { lift(ll::cuCtxCreate_v2(&mut handle, flags, self.handle))? }

        Ok(Context {
            defused: false,
            handle: handle,
        })
    }

    /// Returns the total amount of (non necessarily free) memory, in bytes,
    /// that the device has
    pub fn total_memory(&self) -> Result<usize> {
        let mut bytes = 0;

        unsafe { lift(ll::cuDeviceTotalMem_v2(&mut bytes, self.handle))? };

        Ok(bytes)
    }

    /// Returns maximum number of threads per block
    pub fn max_threads_per_block(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    }

    /// Returns maximum x-dimension of a block
    pub fn max_block_dim_x(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)
    }

    /// Returns maximum y-dimension of a block
    pub fn max_block_dim_y(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y)
    }

    /// Returns maximum z-dimension of a block
    pub fn max_block_dim_z(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z)
    }

    /// Returns maximum x-dimension of a grid
    pub fn max_grid_dim_x(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)
    }

    /// Returns maximum y-dimension of a grid
    pub fn max_grid_dim_y(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)
    }

    /// Returns maximum z-dimension of a grid
    pub fn max_grid_dim_z(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)
    }

    /// Returns maximum amount of shared memory available to a thread block in bytes
    pub fn max_shared_memory_per_block(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
    }

    /// Returns memory available on device for __constant__ variables in a CUDA C kernel in bytes
    pub fn total_constant_memory(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY)
    }

    /// Returns warp size in threads
    pub fn warp_size(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_WARP_SIZE)
    }

    /// Returns maximum pitch in bytes allowed by the memory copy functions that involve memory regions allocated through ::cuMemAllocPitch()
    pub fn max_pitch(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_PITCH)
    }

    /// Returns maximum 1D texture width
    pub fn maximum_texture1d_width(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH)
    }

    /// Returns maximum width for a 1D texture bound to linear memory
    pub fn maximum_texture1d_linear_width(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH)
    }

    /// Returns maximum mipmapped 1D texture width
    pub fn maximum_texture1d_mipmapped_width(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH)
    }

    /// Returns maximum 2D texture width
    pub fn maximum_texture2d_width(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH)
    }

    /// Returns maximum 2D texture height
    pub fn maximum_texture2d_height(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT)
    }

    /// Returns maximum width for a 2D texture bound to linear memory
    pub fn maximum_texture2d_linear_width(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH)
    }

    /// Returns maximum height for a 2D texture bound to linear memory
    pub fn maximum_texture2d_linear_height(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT)
    }

    /// Returns maximum pitch in bytes for a 2D texture bound to linear memory
    pub fn maximum_texture2d_linear_pitch(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH)
    }

    /// Returns maximum mipmapped 2D texture width
    pub fn maximum_texture2d_mipmapped_width(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH)
    }

    /// Returns maximum mipmapped 2D texture height
    pub fn maximum_texture2d_mipmapped_height(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT)
    }

    /// Returns maximum 3D texture width
    pub fn maximum_texture3d_width(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH)
    }

    /// Returns maximum 3D texture height
    pub fn maximum_texture3d_height(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT)
    }

    /// Returns maximum 3D texture depth
    pub fn maximum_texture3d_depth(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH)
    }

    /// Returns alternate maximum 3D texture width, 0 if no alternate maximum 3D texture size is supported
    pub fn maximum_texture3d_width_alternate(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE)
    }

    /// Returns alternate maximum 3D texture height, 0 if no alternate maximum 3D texture size is supported
    pub fn maximum_texture3d_height_alternate(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE)
    }

    /// Returns alternate maximum 3D texture depth, 0 if no alternate maximum 3D texture size is supported
    pub fn maximum_texture3d_depth_alternate(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE)
    }

    /// Returns maximum cubemap texture width or height
    pub fn maximum_texturecubemap_width(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH)
    }

    /// Returns maximum 1D layered texture width
    pub fn maximum_texture1d_layered_width(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH)
    }

    /// Returns maximum layers in a 1D layered texture
    pub fn maximum_texture1d_layered_layers(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS)
    }

    /// Returns maximum 2D layered texture width
    pub fn maximum_texture2d_layered_width(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH)
    }

    /// Returns maximum 2D layered texture height
    pub fn maximum_texture2d_layered_height(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT)
    }

    /// Returns maximum layers in a 2D layered texture
    pub fn maximum_texture2d_layered_layers(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS)
    }

    /// Returns maximum cubemap layered texture width or height
    pub fn maximum_texturecubemap_layered_width(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH)
    }

    /// Returns maximum layers in a cubemap layered texture
    pub fn maximum_texturecubemap_layered_layers(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS)
    }

    /// Returns maximum 1D surface width
    pub fn maximum_surface1d_width(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH)
    }

    /// Returns maximum 2D surface width
    pub fn maximum_surface2d_width(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH)
    }

    /// Returns maximum 2D surface height
    pub fn maximum_surface2d_height(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT)
    }

    /// Returns maximum 3D surface width
    pub fn maximum_surface3d_width(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH)
    }

    /// Returns maximum 3D surface height
    pub fn maximum_surface3d_height(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT)
    }

    /// Returns maximum 3D surface depth
    pub fn maximum_surface3d_depth(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH)
    }

    /// Returns maximum 1D layered surface width
    pub fn maximum_surface1d_layered_width(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH)
    }

    /// Returns maximum layers in a 1D layered surface
    pub fn maximum_surface1d_layered_layers(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS)
    }

    /// Returns maximum 2D layered surface width
    pub fn maximum_surface2d_layered_width(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH)
    }

    /// Returns maximum 2D layered surface height
    pub fn maximum_surface2d_layered_height(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT)
    }

    /// Returns maximum layers in a 2D layered surface
    pub fn maximum_surface2d_layered_layers(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS)
    }

    /// Returns maximum cubemap surface width
    pub fn maximum_surfacecubemap_width(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH)
    }

    /// Returns maximum cubemap layered surface width
    pub fn maximum_surfacecubemap_layered_width(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH)
    }

    /// Returns maximum layers in a cubemap layered surface
    pub fn maximum_surfacecubemap_layered_layers(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS)
    }

    /// Returns maximum number of 32-bit registers available to a thread block
    pub fn max_registers_per_block(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK)
    }

    /// Returns the typical clock frequency in kilohertz
    pub fn clock_rate(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_CLOCK_RATE)
    }

    /// Returns alignment requirement; texture base addresses aligned to ::textureAlign bytes do not need an offset applied to texture fetches
    pub fn texture_alignment(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT)
    }

    /// Returns pitch alignment requirement for 2D texture references bound to pitched memory
    pub fn texture_pitch_alignment(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT)
    }

    /// Returns true if the device can concurrently copy memory between host and device while executing a kernel, or false if not
    pub fn gpu_overlap(&self) -> Result<bool> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_GPU_OVERLAP).map(|f| f != 0)
    }

    /// Returns number of multiprocessors on the device
    pub fn multiprocessor_count(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    }

    /// Returns true if there is a run time limit for kernels executed on the device, or false if not
    pub fn kernel_exec_timeout(&self) -> Result<bool> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT).map(|f| f != 0)
    }

    /// Returns true if the device is integrated with the memory subsystem, or false if not
    pub fn integrated(&self) -> Result<bool> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_INTEGRATED).map(|f| f != 0)
    }

    /// Returns true if the device can map host memory into the CUDA address space, or false if not
    pub fn can_map_host_memory(&self) -> Result<bool> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY).map(|f| f != 0)
    }

    /// Returns compute mode that device is currently in
    pub fn compute_mode(&self) -> Result<ComputeMode> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_MODE).map(|mode| match mode {
            mode if mode == ll::CUcomputemode_enum::CU_COMPUTEMODE_DEFAULT as i32 => ComputeMode::Default,
            mode if mode == ll::CUcomputemode_enum::CU_COMPUTEMODE_PROHIBITED as i32 => ComputeMode::Prohibited,
            mode if mode == ll::CUcomputemode_enum::CU_COMPUTEMODE_EXCLUSIVE_PROCESS as i32 => ComputeMode::ExclusiveProcess,
            mode => panic!("Unknown device compute mode {:?}", mode),
        })
    }

    /// Returns true if the device supports
    ///   executing multiple kernels within the same context simultaneously, or false if
    ///   not. It is not guaranteed that multiple kernels will be resident
    ///   on the device concurrently so this feature should not be relied upon for
    ///   correctness
    pub fn concurrent_kernels(&self) -> Result<bool> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS).map(|f| f != 0)
    }

    /// Returns true if error correction is enabled on the device, false if error correction is disabled or not supported by this device
    pub fn ecc_enabled(&self) -> Result<bool> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_ECC_ENABLED).map(|f| f != 0)
    }

    /// Returns pCI bus identifier of the device
    pub fn pci_bus_id(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_PCI_BUS_ID)
    }

    /// Returns pCI device (also known as slot) identifier of the device
    pub fn pci_device_id(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID)
    }

    /// Returns true if the device is using a TCC driver. TCC is only available on Tesla hardware running Windows Vista or later
    pub fn tcc_driver(&self) -> Result<bool> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_TCC_DRIVER).map(|f| f != 0)
    }

    /// Returns peak memory clock frequency in kilohertz
    pub fn memory_clock_rate(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)
    }

    /// Returns global memory bus width in bits
    pub fn global_memory_bus_width(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)
    }

    /// Returns size of L2 cache in bytes. 0 if the device doesn't have L2 cache
    pub fn l2_cache_size(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
    }

    /// Returns maximum resident threads per multiprocessor;\
    pub fn max_threads_per_multiprocessor(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)
    }

    /// Returns true if the device shares a unified address space with the host, or false if not
    pub fn unified_addressing(&self) -> Result<bool> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING).map(|f| f != 0)
    }

    /// Returns major compute capability version number
    pub fn compute_capability_major(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
    }

    /// Returns minor compute capability version number
    pub fn compute_capability_minor(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
    }

    /// Returns true if device supports caching globals in L1 cache, false if caching globals in L1 cache is not supported by this device
    pub fn global_l1_cache_supported(&self) -> Result<bool> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED).map(|f| f != 0)
    }

    /// Returns true if device supports caching locals in L1 cache, false if caching locals in L1 cache is not supported by this device
    pub fn local_l1_cache_supported(&self) -> Result<bool> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED).map(|f| f != 0)
    }

    /// Returns maximum amount of shared memory available to a multiprocessor in bytes; this amount is shared by all thread blocks simultaneously resident on a multiprocessor
    pub fn max_shared_memory_per_multiprocessor(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)
    }

    /// Returns maximum number of 32-bit registers available to a multiprocessor; this number is shared by all thread blocks simultaneously resident on a multiprocessor
    pub fn max_registers_per_multiprocessor(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR)
    }

    /// Returns true if device supports allocating managed memory on this system, false if allocating managed memory is not supported by the device on this system
    pub fn managed_memory(&self) -> Result<bool> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY).map(|f| f != 0)
    }

    /// Returns true if device is on a multi-GPU board, false if not
    pub fn multi_gpu_board(&self) -> Result<bool> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD).map(|f| f != 0)
    }

    /// Returns unique identifier for a group of devices associated with the same board. Devices on the same multi-GPU board will share the same identifier
    pub fn multi_gpu_board_group_id(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID)
    }

    /// Returns true if link between the device and the host supports native atomic operations, false if not
    pub fn host_native_atomic_supported(&self) -> Result<bool> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED).map(|f| f != 0)
    }

    /// Returns ratio of single precision performance (in floating-point operations per second) to double precision performance
    pub fn single_to_double_precision_perf_ratio(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO)
    }

    /// Returns device suppports coherently accessing pageable memory without calling cudaHostRegister on it
    pub fn pageable_memory_access(&self) -> Result<i32> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS)
    }

    /// Returns true if device can coherently access managed memory concurrently with the CPU, false if not
    pub fn concurrent_managed_access(&self) -> Result<bool> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS).map(|f| f != 0)
    }

    /// Returns true if device supports Compute Preemption, false if not
    pub fn compute_preemption_supported(&self) -> Result<bool> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED).map(|f| f != 0)
    }

    /// Returns true if device can access host registered memory at the same virtual address as the CPU, false if not
    pub fn can_use_host_pointer_for_registered_mem(&self) -> Result<bool> {
        self.get(ll::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM).map(|f| f != 0)
    }

    fn get(&self, attr: ll::CUdevice_attribute) -> Result<i32> {
        let mut value = 0;

        unsafe {
            lift(ll::cuDeviceGetAttribute(&mut value, attr, self.handle))?
        }

        Ok(value)
    }
}

/// Device compute mode
pub enum ComputeMode {
    ///  Default mode - Device is not restricted and can have multiple CUDA contexts present at a single time.
    Default,
    /// Compute-prohibited mode - Device is prohibited from creating new CUDA contexts.
    Prohibited,
    /// Compute-exclusive-process mode - Device can have only one context used by a single process at a time.
    ExclusiveProcess,
}

/// A function that the CUDA device can execute. AKA a "kernel"
pub struct Function<'ctx: 'm, 'm> {
    handle: ll::CUfunction,
    _module: PhantomData<&'m Module<'ctx>>,
}

impl<'ctx, 'm> Function<'ctx, 'm> {
    /// Execute a function on the GPU
    ///
    /// NOTE This function blocks until the GPU has finished executing the
    /// kernel
    pub fn launch(&self,
                  args: &[&Any],
                  grid: Grid,
                  block: Block)
                  -> Result<()> {
        let stream = Stream::new()?;
        // TODO expose
        let shared_mem_bytes = 0;
        // TODO expose
        let extra = ptr::null_mut();

        unsafe {
            lift(ll::cuLaunchKernel(self.handle,
                                    grid.x,
                                    grid.y,
                                    grid.z,
                                    block.x,
                                    block.y,
                                    block.z,
                                    shared_mem_bytes,
                                    stream.handle,
                                    args.as_ptr() as *mut _,
                                    extra))?
        }

        stream.sync()?;
        stream.destroy()
    }

    /// Returns the maximum number of threads
    ///   per block, beyond which a launch of the function would fail. This number
    ///   depends on both the function and the device on which the function is
    ///   currently loaded.
    pub fn max_threads_per_block(&self) -> Result<i32> {
        self.get(ll::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    }

    /// Returns the size in bytes of
    ///   statically-allocated shared memory per block required by this function.
    ///   This does not include dynamically-allocated shared memory requested by
    ///   the user at runtime.
    pub fn shared_size_bytes(&self) -> Result<i32> {
        self.get(ll::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)
    }

    /// Returns the size in bytes of user-allocated
    ///   constant memory required by this function.
    pub fn const_size_bytes(&self) -> Result<i32> {
        self.get(ll::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES)
    }

    /// Returns the size in bytes of local memory
    ///   used by each thread of this function.
    pub fn local_size_bytes(&self) -> Result<i32> {
        self.get(ll::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)
    }

    /// Returns the number of registers used by each thread
    ///   of this function.
    pub fn num_regs(&self) -> Result<i32> {
        self.get(ll::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_NUM_REGS)
    }

    /// Returns the PTX virtual architecture version for
    ///   which the function was compiled. This value is the major PTX version * 10
    ///   + the minor PTX version, so a PTX version 1.3 function would return the
    ///   value 13. Note that this may return the undefined value of 0 for cubins
    ///   compiled prior to CUDA 3.0.
    pub fn ptx_version(&self) -> Result<i32> {
        self.get(ll::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_PTX_VERSION)
    }

    /// Returns the binary architecture version for
    ///   which the function was compiled. This value is the major binary
    ///   version * 10 + the minor binary version, so a binary version 1.3 function
    ///   would return the value 13. Note that this will return a value of 10 for
    ///   legacy cubins that do not have a properly-encoded binary architecture
    ///   version.
    pub fn binary_version(&self) -> Result<i32> {
        self.get(ll::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_BINARY_VERSION)
    }

    /// Returns the attribute to indicate whether the function has
    ///   been compiled with user specified option "-Xptxas --dlcm=ca" set.
    pub fn cache_mode_ca(&self) -> Result<bool> {
        self.get(ll::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_CACHE_MODE_CA).map(|f| f != 0)
    }

    fn get(&self, attr: ll::CUfunction_attribute) -> Result<i32> {
        let mut value = 0;

        unsafe {
            lift(ll::cuFuncGetAttribute(&mut value, attr, self.handle))?
        }

        Ok(value)
    }
}

/// A CUDA "grid"
pub struct Grid {
    /// The width of the grid in blocks
    pub x: u32,
    /// The height of the grid in blocks
    pub y: u32,
    /// The depth of the grid in blocks
    pub z: u32,
}

impl Grid {
    /// One dimensional grid
    pub fn x(x: u32) -> Self {
        Grid { x: x, y: 1, z: 1 }
    }

    /// Two dimensional grid
    pub fn xy(x: u32, y: u32) -> Self {
        Grid { x: x, y: y, z: 1 }
    }

    /// Three dimensional grid
    pub fn xyz(x: u32, y: u32, z: u32) -> Self {
        Grid { x: x, y: y, z: z }
    }
}

/// A PTX module
pub struct Module<'ctx> {
    handle: ll::CUmodule,
    _context: PhantomData<&'ctx Context>,
}

impl<'ctx> Module<'ctx> {
    /// Retrieves a function from the PTX module
    pub fn function<'m>(&'m self, name: &CStr) -> Result<Function<'ctx, 'm>> {
        let mut handle = ptr::null_mut();

        unsafe {
            lift(ll::cuModuleGetFunction(&mut handle,
                                         self.handle,
                                         name.as_ptr()))?
        }

        Ok(Function {
            handle: handle,
            _module: PhantomData,
        })
    }
}

impl<'ctx> Drop for Module<'ctx> {
    fn drop(&mut self) {
        let _ignored = unsafe { lift(ll::cuModuleUnload(self.handle)) };
    }
}

// TODO expose
struct Stream {
    handle: ll::CUstream,
}

impl Stream {
    fn new() -> Result<Self> {
        let mut handle = ptr::null_mut();
        // TODO expose
        let flags = 0;

        unsafe { lift(ll::cuStreamCreate(&mut handle, flags))? }

        Ok(Stream { handle: handle })
    }

    fn destroy(self) -> Result<()> {
        unsafe { lift(ll::cuStreamDestroy_v2(self.handle)) }
    }

    fn sync(&self) -> Result<()> {
        unsafe { lift(ll::cuStreamSynchronize(self.handle)) }
    }
}

/// Value who's type has been erased
pub enum Any {}

/// Erase the type of a value
#[allow(non_snake_case)]
pub fn Any<T>(ref_: &T) -> &Any {
    unsafe { &*(ref_ as *const T as *const Any) }
}

/// `memcpy` direction
pub enum Direction {
    /// `src` points to device memory. `dst` points to host memory
    DeviceToHost,
    /// `src` points to host memory. `dst` points to device memory
    HostToDevice,
}

#[allow(missing_docs)]
#[derive(Debug)]
pub enum Error {
    AlreadyAcquired,
    AlreadyMapped,
    ArrayIsMapped,
    Assert,
    ContextAlreadyCurrent,
    ContextAlreadyInUse,
    ContextIsDestroyed,
    Deinitialized,
    EccUncorrectable,
    FileNotFound,
    HardwareStackError,
    HostMemoryAlreadyRegistered,
    HostMemoryNotRegistered,
    IllegalAddress,
    IllegalInstruction,
    InvalidAddressSpace,
    InvalidContext,
    InvalidDevice,
    InvalidGraphicsContext,
    InvalidHandle,
    InvalidImage,
    InvalidPc,
    InvalidPtx,
    InvalidSource,
    InvalidValue,
    LaunchFailed,
    LaunchIncompatibleTexturing,
    LaunchOutOfResources,
    LaunchTimeout,
    MapFailed,
    MisalignedAddress,
    NoBinaryForGpu,
    NoDevice,
    NotFound,
    NotInitialized,
    NotMapped,
    NotMappedAsArray,
    NotMappedAsPointer,
    NotPermitted,
    NotReady,
    NotSupported,
    NvlinkUncorrectable,
    OperatingSystem,
    OutOfMemory,
    PeerAccessAlreadyEnabled,
    PeerAccessNotEnabled,
    PeerAccessUnsupported,
    PrimaryContextActive,
    ProfilerAlreadyStarted,
    ProfilerAlreadyStopped,
    ProfilerDisabled,
    ProfilerNotInitialized,
    SharedObjectInitFailed,
    SharedObjectSymbolNotFound,
    TooManyPeers,
    Unknown,
    UnmapFailed,
    UnsupportedLimit,
}

// TODO should this be a method of `Context`?
/// Allocate `n` bytes of memory on the device
pub unsafe fn allocate(n: usize) -> Result<*mut u8> {
    let mut d_ptr = 0;

    lift(ll::cuMemAlloc_v2(&mut d_ptr, n))?;

    Ok(d_ptr as *mut u8)
}

/// Copy `n` bytes of memory from `src` to `dst`
///
/// `direction` indicates where `src` and `dst` are located (device or host)
pub unsafe fn copy<T>(src: *const T,
                      dst: *mut T,
                      count: usize,
                      direction: Direction)
                      -> Result<()> {
    use self::Direction::*;

    let bytes = count * mem::size_of::<T>();

    lift(match direction {
        DeviceToHost => ll::cuMemcpyDtoH_v2(dst as *mut _, src as u64, bytes),
        HostToDevice => ll::cuMemcpyHtoD_v2(dst as u64, src as *const _, bytes),
    })?;

    Ok(())
}

// TODO same question as `allocate`
/// Free the memory pointed to by `ptr`
pub unsafe fn deallocate(ptr: *mut u8) -> Result<()> {
    lift(ll::cuMemFree_v2(ptr as u64))
}

/// Initialize the CUDA runtime
pub fn initialize() -> Result<()> {
    // TODO expose
    let flags = 0;

    unsafe { lift(ll::cuInit(flags)) }
}

/// Returns the version of the CUDA runtime
pub fn version() -> Result<i32> {
    let mut version = 0;

    unsafe { lift(ll::cuDriverGetVersion(&mut version))? }

    Ok(version)
}

#[allow(missing_docs)]
pub type Result<T> = result::Result<T, Error>;

fn lift(e: ll::CUresult) -> Result<()> {
    use self::Error::*;
    use self::ll::cudaError_enum::*;

    Err(match e {
        CUDA_SUCCESS => return Ok(()),
        CUDA_ERROR_ALREADY_ACQUIRED => AlreadyAcquired,
        CUDA_ERROR_ALREADY_MAPPED => AlreadyMapped,
        CUDA_ERROR_ARRAY_IS_MAPPED => ArrayIsMapped,
        CUDA_ERROR_ASSERT => Assert,
        CUDA_ERROR_CONTEXT_ALREADY_CURRENT => ContextAlreadyCurrent,
        CUDA_ERROR_CONTEXT_ALREADY_IN_USE => ContextAlreadyInUse,
        CUDA_ERROR_CONTEXT_IS_DESTROYED => ContextIsDestroyed,
        CUDA_ERROR_DEINITIALIZED => Deinitialized,
        CUDA_ERROR_ECC_UNCORRECTABLE => EccUncorrectable,
        CUDA_ERROR_FILE_NOT_FOUND => FileNotFound,
        CUDA_ERROR_HARDWARE_STACK_ERROR => HardwareStackError,
        CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED => {
            HostMemoryAlreadyRegistered
        }
        CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED => HostMemoryNotRegistered,
        CUDA_ERROR_ILLEGAL_ADDRESS => IllegalAddress,
        CUDA_ERROR_ILLEGAL_INSTRUCTION => IllegalInstruction,
        CUDA_ERROR_INVALID_ADDRESS_SPACE => InvalidAddressSpace,
        CUDA_ERROR_INVALID_CONTEXT => InvalidContext,
        CUDA_ERROR_INVALID_DEVICE => InvalidDevice,
        CUDA_ERROR_INVALID_GRAPHICS_CONTEXT => InvalidGraphicsContext,
        CUDA_ERROR_INVALID_HANDLE => InvalidHandle,
        CUDA_ERROR_INVALID_IMAGE => InvalidImage,
        CUDA_ERROR_INVALID_PC => InvalidPc,
        CUDA_ERROR_INVALID_PTX => InvalidPtx,
        CUDA_ERROR_INVALID_SOURCE => InvalidSource,
        CUDA_ERROR_INVALID_VALUE => InvalidValue,
        CUDA_ERROR_LAUNCH_FAILED => LaunchFailed,
        CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING => LaunchIncompatibleTexturing,
        CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES => LaunchOutOfResources,
        CUDA_ERROR_LAUNCH_TIMEOUT => LaunchTimeout,
        CUDA_ERROR_MAP_FAILED => MapFailed,
        CUDA_ERROR_MISALIGNED_ADDRESS => MisalignedAddress,
        CUDA_ERROR_NOT_FOUND => NotFound,
        CUDA_ERROR_NOT_INITIALIZED => NotInitialized,
        CUDA_ERROR_NOT_MAPPED => NotMapped,
        CUDA_ERROR_NOT_MAPPED_AS_ARRAY => NotMappedAsArray,
        CUDA_ERROR_NOT_MAPPED_AS_POINTER => NotMappedAsPointer,
        CUDA_ERROR_NOT_PERMITTED => NotPermitted,
        CUDA_ERROR_NOT_READY => NotReady,
        CUDA_ERROR_NOT_SUPPORTED => NotSupported,
        CUDA_ERROR_NO_BINARY_FOR_GPU => NoBinaryForGpu,
        CUDA_ERROR_NO_DEVICE => NoDevice,
        CUDA_ERROR_OPERATING_SYSTEM => OperatingSystem,
        CUDA_ERROR_OUT_OF_MEMORY => OutOfMemory,
        CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED => PeerAccessAlreadyEnabled,
        CUDA_ERROR_PEER_ACCESS_NOT_ENABLED => PeerAccessNotEnabled,
        CUDA_ERROR_PEER_ACCESS_UNSUPPORTED => PeerAccessUnsupported,
        CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE => PrimaryContextActive,
        CUDA_ERROR_PROFILER_ALREADY_STARTED => ProfilerAlreadyStarted,
        CUDA_ERROR_PROFILER_ALREADY_STOPPED => ProfilerAlreadyStopped,
        CUDA_ERROR_PROFILER_DISABLED => ProfilerDisabled,
        CUDA_ERROR_PROFILER_NOT_INITIALIZED => ProfilerNotInitialized,
        CUDA_ERROR_SHARED_OBJECT_INIT_FAILED => SharedObjectInitFailed,
        CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND => SharedObjectSymbolNotFound,
        CUDA_ERROR_TOO_MANY_PEERS => TooManyPeers,
        CUDA_ERROR_UNKNOWN => Unknown,
        CUDA_ERROR_UNMAP_FAILED => UnmapFailed,
        CUDA_ERROR_UNSUPPORTED_LIMIT => UnsupportedLimit,
        CUDA_ERROR_NVLINK_UNCORRECTABLE => NvlinkUncorrectable,
    })
}
