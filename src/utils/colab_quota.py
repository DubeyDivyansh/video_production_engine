"""
Colab Resource Quota Monitor & Account Switchover Trigger.
Provides automated detection of Colab GPU quota exhaustion, calculates
multi-account capacity multipliers, and outputs actionable switchover instructions.
"""

from typing import Dict, Any, Optional
import subprocess
import sys


class ColabQuotaMonitor:
    """
    Monitors GPU availability and triggers multi-account switchover
    when Colab daily GPU quotas are exhausted.
    """

    GPU_CAPACITY_PER_ACCOUNT_HOURS = 4.5
    VIDEOS_PER_ACCOUNT_ESTIMATE = 28  # ~22-second master videos per account/day

    @classmethod
    def calculate_capacity(cls, num_accounts: int = 1) -> Dict[str, Any]:
        """Calculates compute capacity and video output yield for N accounts."""
        num = max(1, num_accounts)
        gpu_hours = num * cls.GPU_CAPACITY_PER_ACCOUNT_HOURS
        videos = num * cls.VIDEOS_PER_ACCOUNT_ESTIMATE
        return {
            "num_accounts": num,
            "multiplier": f"{num}x",
            "daily_gpu_hours": gpu_hours,
            "estimated_daily_master_videos": videos,
        }

    @classmethod
    def is_quota_exhausted(cls, error_message: str = "") -> bool:
        """Checks if a given error or runtime failure indicates Colab GPU quota depletion."""
        exhaustion_indicators = [
            "cannot connect to gpu backend",
            "usage limits in colab",
            "resource limit reached",
            "no cuda gpus are available",
            "cuda driver initialization failed",
            "gpu backend not available",
            "rate limit exceeded",
        ]
        msg = error_message.lower()
        return any(ind in msg for ind in exhaustion_indicators)

    @classmethod
    def trigger_account_switchover(cls, current_account: str = "Account A", target_account: str = "Account B") -> str:
        """Generates a structured switchover prompt instructing the user to transition to the next account."""
        prompt = f"""
================================================================================
🚨 [COLAB GPU QUOTA EXHAUSTION DETECTED ON {current_account.upper()}]
================================================================================
• Status: Daily free GPU allocation has been reached on {current_account}.
• Cooldown: Initiated on {current_account} (auto-resets in 12–18 hours).

🔄 ACTION REQUIRED: SWITCH TO {target_account.upper()} FOR 2X COMPUTE CONTINUITY
--------------------------------------------------------------------------------
1. Open an alternate browser profile or incognito tab logged into {target_account}.
2. Navigate to quick_demo.ipynb:
   👉 https://colab.research.google.com/github/DubeyDivyansh/DubeyDivyansh/blob/main/quick_demo.ipynb
3. Connect to a T4 GPU runtime.
4. Run Section 1 (Mount Google Drive).
   - Because '/MyDrive/video_gen_platform' is shared as a shortcut with {target_account},
     all cached weights, avatar assets, and previous outputs mount in < 5 seconds.
5. Resume your video production queue seamlessly with 0 redundant downloads!
================================================================================
"""
        print(prompt)
        return prompt
