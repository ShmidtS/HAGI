#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoreBlend {
    pub dot_weight: f32,
    pub truth_weight: f32,
    pub recency_weight: f32,
}

impl Default for ScoreBlend {
    fn default() -> Self {
        Self {
            dot_weight: 0.6,
            truth_weight: 0.3,
            recency_weight: 0.1,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum NarsRoutePolicy {
    FixedTopK {
        top_k: usize,
        blend: ScoreBlend,
    },
    ConfidenceThreshold {
        min_k: usize,
        max_k: usize,
        cumulative_confidence: f32,
        blend: ScoreBlend,
    },
}

impl Default for NarsRoutePolicy {
    fn default() -> Self {
        Self::FixedTopK {
            top_k: 1,
            blend: ScoreBlend::default(),
        }
    }
}

impl NarsRoutePolicy {
    pub fn blend(&self) -> &ScoreBlend {
        match self {
            Self::FixedTopK { blend, .. } | Self::ConfidenceThreshold { blend, .. } => blend,
        }
    }

    pub fn max_candidates(&self) -> usize {
        match self {
            Self::FixedTopK { top_k, .. } => *top_k,
            Self::ConfidenceThreshold { max_k, .. } => *max_k,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_blend_uses_nars_scores() {
        assert_eq!(
            ScoreBlend::default(),
            ScoreBlend {
                dot_weight: 0.6,
                truth_weight: 0.3,
                recency_weight: 0.1,
            }
        );
    }

    #[test]
    fn default_policy_uses_fixed_top_k_with_blended_weights() {
        assert_eq!(
            NarsRoutePolicy::default(),
            NarsRoutePolicy::FixedTopK {
                top_k: 1,
                blend: ScoreBlend::default(),
            }
        );
    }

    #[test]
    fn policy_exposes_candidate_limit() {
        let blend = ScoreBlend::default();
        assert_eq!(
            NarsRoutePolicy::FixedTopK { top_k: 3, blend }.max_candidates(),
            3
        );
        assert_eq!(
            NarsRoutePolicy::ConfidenceThreshold {
                min_k: 1,
                max_k: 4,
                cumulative_confidence: 0.8,
                blend,
            }
            .max_candidates(),
            4
        );
    }
}
