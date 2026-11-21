//! Execute one compute dispatch against an initialized buffer using software D3D12.

use std::mem::{size_of_val, ManuallyDrop};
use std::os::windows::io::{FromRawHandle, OwnedHandle};
use std::{ptr, slice};
use windows::core::Interface;
use windows::Win32::Foundation::WAIT_OBJECT_0;
use windows::Win32::Graphics::Direct3D::{ID3DBlob, D3D_FEATURE_LEVEL_11_0};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory2, IDXGIAdapter, IDXGIFactory4, DXGI_CREATE_FACTORY_FLAGS,
};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

pub(super) fn dispatch_buffer(
    dxil: &[u8],
    input: [u32; 4],
) -> Result<[u32; 4], Box<dyn std::error::Error>> {
    // SAFETY: COM objects stay alive through fence completion. The shader and the
    // root signature agree on a structured UAV at u0 with one u32 per element.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
        let adapter: IDXGIAdapter = factory.EnumWarpAdapter()?;
        let mut device = None;
        D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut device)?;
        let device: ID3D12Device = device.expect("D3D12 device");

        let parameter = D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR::default(),
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        };
        let desc = D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: 1,
            pParameters: &parameter,
            ..Default::default()
        };
        let mut blob: Option<ID3DBlob> = None;
        D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, None)?;
        let blob = blob.expect("serialized root signature");
        let root_bytes =
            slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, root_bytes)?;
        let mut desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: ManuallyDrop::new(Some(root.clone())),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: dxil.as_ptr().cast(),
                BytecodeLength: dxil.len(),
            },
            ..Default::default()
        };
        let pipeline = device.CreateComputePipelineState::<ID3D12PipelineState>(&desc);
        ManuallyDrop::drop(&mut desc.pRootSignature);
        let pipeline = pipeline?;

        let kind = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: kind,
            ..Default::default()
        })?;
        let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(kind)?;
        let list: ID3D12GraphicsCommandList =
            device.CreateCommandList(0, kind, &allocator, &pipeline)?;
        let bytes = size_of_val(&input);
        let none = D3D12_RANGE { Begin: 0, End: 0 };
        let all = D3D12_RANGE {
            Begin: 0,
            End: bytes,
        };

        let upload = buffer(
            &device,
            bytes,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            D3D12_RESOURCE_FLAG_NONE,
        )?;
        let mut mapped = ptr::null_mut();
        upload.Map(0, Some(&none), Some(&mut mapped))?;
        ptr::copy_nonoverlapping(input.as_ptr(), mapped.cast::<u32>(), input.len());
        upload.Unmap(0, Some(&all));
        let data = buffer(
            &device,
            bytes,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        )?;
        let readback = buffer(
            &device,
            bytes,
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_FLAG_NONE,
        )?;

        list.CopyBufferRegion(&data, 0, &upload, 0, bytes as u64);
        transition(
            &list,
            &data,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        list.SetComputeRootSignature(&root);
        list.SetComputeRootUnorderedAccessView(0, data.GetGPUVirtualAddress());
        list.Dispatch(1, 1, 1);
        transition(
            &list,
            &data,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        );
        list.CopyBufferRegion(&readback, 0, &data, 0, bytes as u64);
        list.Close()?;

        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        let event = CreateEventW(None, false, false, None)?;
        let _event_handle = OwnedHandle::from_raw_handle(event.0);
        let command_list = list.cast::<ID3D12CommandList>()?;
        queue.ExecuteCommandLists(&[Some(command_list)]);
        queue.Signal(&fence, 1)?;
        fence.SetEventOnCompletion(1, event)?;
        let status = WaitForSingleObject(event, 10_000);
        if status != WAIT_OBJECT_0 {
            return Err(format!("D3D12 fence wait returned {:#x}", status.0).into());
        }
        device.GetDeviceRemovedReason()?;

        let mut output = [0u32; 4];
        readback.Map(0, Some(&all), Some(&mut mapped))?;
        ptr::copy_nonoverlapping(mapped.cast::<u32>(), output.as_mut_ptr(), output.len());
        readback.Unmap(0, Some(&none));
        Ok(output)
    }
}

unsafe fn buffer(
    device: &ID3D12Device,
    bytes: usize,
    heap_type: D3D12_HEAP_TYPE,
    state: D3D12_RESOURCE_STATES,
    flags: D3D12_RESOURCE_FLAGS,
) -> windows::core::Result<ID3D12Resource> {
    let heap = D3D12_HEAP_PROPERTIES {
        Type: heap_type,
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
        ..Default::default()
    };
    let mut desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes as u64,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    };
    desc.SampleDesc.Count = 1;
    let mut resource = None;
    device.CreateCommittedResource(
        &heap,
        D3D12_HEAP_FLAG_NONE,
        &desc,
        state,
        None,
        &mut resource,
    )?;
    Ok(resource.expect("committed buffer"))
}

unsafe fn transition(
    list: &ID3D12GraphicsCommandList,
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) {
    let mut barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: ManuallyDrop::new(Some(resource.clone())),
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    };
    list.ResourceBarrier(slice::from_ref(&barrier));
    let mut transition = ManuallyDrop::take(&mut barrier.Anonymous.Transition);
    ManuallyDrop::drop(&mut transition.pResource);
}
