#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CycleIndex(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HCycle(pub CycleIndex);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LCycle(pub CycleIndex);
