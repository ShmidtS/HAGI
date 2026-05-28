//! MSA Sparse Memory Adapter — top-k memory routing over registered slots.

pub mod attention_bridge;
pub mod fetch;
pub mod interleave;
pub mod kv_cache;
pub mod registry;
pub mod rope;
pub mod route;
pub mod slot;
pub mod sparse_attention;

pub use attention_bridge::sparse_attention_with_memory;
pub use fetch::{fetch_pages, FetchEvent};
pub use interleave::{
    run_memory_interleave, MemoryInterleaveConfig, MemoryInterleaveReport, MemoryStopReason,
};
pub use kv_cache::{HostKvCache, HostKvPage, KVCache};
pub use registry::SlotRegistry;
pub use rope::{apply_document_wise_rope, DocumentWiseRoPE, RoPEPosition};
pub use route::{
    route_from_hdim_invariant, route_top_k, routing_query_from_invariant, Cl3, MsaConfig, MsaError,
    RouteSelection, RoutingQueryView, SparseRouter,
};
pub use slot::MemorySlot;
pub use sparse_attention::{
    sparse_attention_over_pages, sparse_attention_with_local_context, SparseAttention,
};
