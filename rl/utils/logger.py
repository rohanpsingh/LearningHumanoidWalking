"""Training logger for RL algorithms.

Provides a unified interface for logging training metrics to TensorBoard.
"""

from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class TrainingLogger:
    """Handles logging of training metrics to TensorBoard.

    Extracts logging responsibility from algorithm classes to provide
    a reusable, testable logging interface.
    """

    def __init__(self, log_dir: str | Path, flush_secs: int = 10):
        """Initialize the training logger.

        Args:
            log_dir: Directory to save TensorBoard logs.
            flush_secs: How often to flush logs to disk.
        """
        self.log_dir = Path(log_dir)
        self._writer = SummaryWriter(log_dir=str(self.log_dir), flush_secs=flush_secs)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a single scalar value.

        Args:
            tag: Name of the metric (e.g., "Loss/actor").
            value: Scalar value to log.
            step: Training step/iteration.
        """
        self._writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, values: dict[str, float], step: int) -> None:
        """Log multiple related scalars.

        Args:
            main_tag: Group name for the metrics.
            values: Dictionary mapping metric names to values.
            step: Training step/iteration.
        """
        self._writer.add_scalars(main_tag, values, step)

    def log_training_metrics(
        self,
        actor_loss: float,
        critic_loss: float,
        mirror_loss: float,
        imitation_loss: float,
        mean_reward: float,
        mean_ep_len: float,
        mean_noise_std: float,
        step: int,
    ) -> None:
        """Log standard training metrics.

        Args:
            actor_loss: Actor network loss.
            critic_loss: Critic network loss.
            mirror_loss: Mirror symmetry loss.
            imitation_loss: Imitation learning loss.
            mean_reward: Mean episode reward.
            mean_ep_len: Mean episode length.
            mean_noise_std: Mean action noise standard deviation.
            step: Training step/iteration.
        """
        self._writer.add_scalar("Loss/actor", actor_loss, step)
        self._writer.add_scalar("Loss/critic", critic_loss, step)
        self._writer.add_scalar("Loss/mirror", mirror_loss, step)
        self._writer.add_scalar("Loss/imitation", imitation_loss, step)
        self._writer.add_scalar("Train/mean_reward", mean_reward, step)
        self._writer.add_scalar("Train/mean_episode_length", mean_ep_len, step)
        self._writer.add_scalar("Train/mean_noise_std", mean_noise_std, step)

    def log_eval_metrics(
        self,
        mean_reward: float,
        mean_ep_len: float,
        step: int,
    ) -> None:
        """Log evaluation metrics.

        Args:
            mean_reward: Mean evaluation episode reward.
            mean_ep_len: Mean evaluation episode length.
            step: Training step/iteration.
        """
        self._writer.add_scalar("Eval/mean_reward", mean_reward, step)
        self._writer.add_scalar("Eval/mean_episode_length", mean_ep_len, step)

    def log_timing_metrics(
        self,
        fps: float,
        sample_time: float,
        optimize_time: float,
        total_time: float,
        step: int,
    ) -> None:
        """Log timing/performance metrics.

        Args:
            fps: Frames (steps) per second.
            sample_time: Time spent sampling trajectories (seconds).
            optimize_time: Time spent in optimizer (seconds).
            total_time: Total elapsed training time (seconds).
            step: Training step/iteration.
        """
        self._writer.add_scalar("Time/fps", fps, step)
        self._writer.add_scalar("Time/sample_time", sample_time, step)
        self._writer.add_scalar("Time/optimize_time", optimize_time, step)
        self._writer.add_scalar("Time/total_elapsed", total_time, step)

    def flush(self) -> None:
        """Flush pending logs to disk."""
        self._writer.flush()

    def close(self) -> None:
        """Close the logger and release resources."""
        self._writer.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures logger is closed."""
        self.close()
        return False
