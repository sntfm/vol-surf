use shared_memory::{ShmemConf, Shmem};
use std::sync::atomic::{AtomicU64, Ordering};
use serde_json::Value;

use crate::error::{Result, ConnectorError};

const SHMEM_NAME: &str = "polygon_options_data";
const SHMEM_SIZE: usize = 1024 * 1024 * 10; // 10MB

pub struct SharedMemManager {
    shmem: Shmem,
    write_pos: AtomicU64,
}

impl SharedMemManager {
    pub fn new() -> Result<Self> {
        let shmem = ShmemConf::new()
            .size(SHMEM_SIZE)
            .os_id(SHMEM_NAME)
            .create()
            .map_err(|e| ConnectorError::SharedMemoryError(e.to_string()))?;
            
        Ok(Self {
            shmem,
            write_pos: AtomicU64::new(0),
        })
    }
    
    pub fn update_data(&mut self, data: &Value) -> Result<()> {
        // For now, we'll just store the raw JSON as bytes
        // TODO: Implement FlatBuffer serialization once schema is generated
        let json_bytes = serde_json::to_vec(data)
            .map_err(ConnectorError::SerdeError)?;
            
        let data_len = json_bytes.len();
        
        // Get a mutable slice of the shared memory
        let mem_slice = unsafe { self.shmem.as_slice_mut() };
        
        // Calculate the write position with wrapping
        let current_pos = self.write_pos.load(Ordering::Relaxed) as usize;
        let new_pos = (current_pos + data_len) % SHMEM_SIZE;
        
        // Write the data length first (4 bytes)
        if current_pos + 4 <= SHMEM_SIZE {
            mem_slice[current_pos..current_pos + 4].copy_from_slice(&(data_len as u32).to_le_bytes());
        } else {
            // Handle wrapping for length
            let first_part = SHMEM_SIZE - current_pos;
            mem_slice[current_pos..].copy_from_slice(&(data_len as u32).to_le_bytes()[..first_part]);
            mem_slice[..4 - first_part].copy_from_slice(&(data_len as u32).to_le_bytes()[first_part..]);
        }
        
        // Write the actual data
        let write_start = (current_pos + 4) % SHMEM_SIZE;
        if write_start + data_len <= SHMEM_SIZE {
            mem_slice[write_start..write_start + data_len].copy_from_slice(&json_bytes);
        } else {
            // Handle wrapping for data
            let first_part = SHMEM_SIZE - write_start;
            mem_slice[write_start..].copy_from_slice(&json_bytes[..first_part]);
            mem_slice[..data_len - first_part].copy_from_slice(&json_bytes[first_part..]);
        }
        
        // Update the write position
        self.write_pos.store(new_pos as u64, Ordering::Release);
        
        Ok(())
    }
} 