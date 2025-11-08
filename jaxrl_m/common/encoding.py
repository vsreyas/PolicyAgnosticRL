from typing import Dict, Optional, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange, repeat
from typing import Dict, Tuple


class EncodingWrapper(nn.Module):
    """
    Encodes observations into a single flat encoding, adding additional
    functionality for adding proprioception and stopping the gradient.

    Args:
        encoder: The encoder network.
        use_proprio: Whether to concatenate proprioception (after encoding).
        stop_gradient: Whether to stop the gradient after the encoder.
    """

    encoder: nn.Module
    use_proprio: bool
    stop_gradient: bool
    proprioceptive_dims: Optional[int] = None
    enable_stacking: bool = False

    def __call__(
        self, observations: Dict[str, jnp.ndarray], train: bool
    ) -> jnp.ndarray:
        # import pdb; pdb.set_trace()
        if (
            isinstance(observations, flax.core.FrozenDict)
            or isinstance(observations, dict)
            and ("image" in observations or "encoding" in observations)
        ):
            if "encoding" in observations:
                return observations["encoding"]
            obs = observations["image"]
            if self.enable_stacking:
                # Combine stacking and channels into a single dimension
                if len(obs.shape) == 4:
                    obs = rearrange(obs, "T H W C -> H W (T C)")
                if len(obs.shape) == 5:
                    obs = rearrange(obs, "B T H W C -> B H W (T C)")

        else:
            obs = observations

        encoding = self.encoder(obs, train=train)

        if self.use_proprio:
            proprio = observations["proprio"]
            if self.proprioceptive_dims is not None:
                # proprio = proprio[..., -self.proprioceptive_dims :]
                proprio = proprio[..., : self.proprioceptive_dims :]
            encoding = jnp.concatenate([encoding, proprio], axis=-1)
        if self.stop_gradient:
            encoding = jax.lax.stop_gradient(encoding)
        return encoding


class GCEncodingWrapper(nn.Module):
    """
    Encodes observations and goals into a single flat encoding. Handles all the
    logic about when/how to combine observations and goals.

    Takes a tuple (observations, goals) as input.

    Args:
        encoder: The encoder network for observations.
        goal_encoder: The encoder to use for goals (optional). If None, early
            goal concatenation is used, i.e. the goal is concatenated to the
            observation channel-wise before passing it through the encoder.
        use_proprio: Whether to concatenate proprioception (after encoding).
        stop_gradient: Whether to stop the gradient after the encoder.
    """

    encoder: nn.Module
    goal_encoder: Optional[nn.Module]
    use_proprio: bool
    stop_gradient: bool

    def __call__(
        self,
        observations_and_goals: Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]],
        train: bool,
    ) -> jnp.ndarray:
        if (
            isinstance(observations_and_goals, dict)
            and "encoding" in observations_and_goals
        ):
            return observations_and_goals["encoding"]

        observations, goals = observations_and_goals

        if "encoding" in observations:
            return observations["encoding"]

        if "image" not in observations and "image" not in goals:
            if len(observations.shape) == 3:
                obs_horizon = observations.shape[1]
                goals = repeat(goals, "B E -> B repeat E", repeat=obs_horizon)
            return self.encoder(
                jnp.concatenate([observations, goals], axis=-1), train=train
            )

        if len(observations["image"].shape) == 5:
            # obs history case
            batch_size, obs_horizon = observations["image"].shape[:2]
            # fold batch_size into obs_horizon to encode each frame separately
            obs_image = rearrange(observations["image"], "B T H W C -> (B T) H W C")
            # repeat goals so that there's a goal for each frame
            goal_image = repeat(
                goals["image"], "B H W C -> (B repeat) H W C", repeat=obs_horizon
            )
        else:
            obs_image = observations["image"]
            goal_image = goals["image"]

        if self.goal_encoder is None:
            # early goal concat
            encoder_inputs = jnp.concatenate([obs_image, goal_image], axis=-1)
            encoding = self.encoder(encoder_inputs, train=train)
        else:
            # late fusion
            encoding = self.encoder(obs_image, train=train)
            goal_encoding = self.goal_encoder(goals["image"], train=train)
            encoding = jnp.concatenate([encoding, goal_encoding], axis=-1)

        if len(observations["image"].shape) == 5:
            # unfold obs_horizon from batch_size
            encoding = rearrange(
                encoding, "(B T) F -> B (T F)", B=batch_size, T=obs_horizon
            )

        if self.use_proprio:
            encoding = jnp.concatenate([encoding, observations["proprio"]], axis=-1)

        if self.stop_gradient:
            encoding = jax.lax.stop_gradient(encoding)

        return encoding


class LCEncodingWrapper(nn.Module):
    """
    Encodes observations and language instructions into a single flat encoding.

    Takes a tuple (observations, goals) as input, where goals contains the language instruction.

    Args:
        encoder: The encoder network for observations.
        use_proprio: Whether to concatenate proprioception (after encoding).
        stop_gradient: Whether to stop the gradient after the encoder.
    """

    encoder: nn.Module
    use_proprio: bool
    stop_gradient: bool

    def __call__(
        self,
        observations_and_goals: Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]],
    ) -> jnp.ndarray:
        observations, goals = observations_and_goals

        if len(observations["image"].shape) == 5:
            # obs history case
            batch_size, obs_horizon = observations["image"].shape[:2]
            # fold batch_size into obs_horizon to encode each frame separately
            obs_image = rearrange(observations["image"], "B T H W C -> (B T) H W C")
            # repeat language so that there's an instruction for each frame
            language = repeat(
                goals["language"], "B E -> (B repeat) E", repeat=obs_horizon
            )
        else:
            obs_image = observations["image"]
            language = goals["language"]

        encoding = self.encoder(obs_image, cond_var=language)

        if len(observations["image"].shape) == 5:
            # unfold obs_horizon from batch_size
            encoding = rearrange(
                encoding, "(B T) F -> B (T F)", B=batch_size, T=obs_horizon
            )

        if self.use_proprio:
            encoding = jnp.concatenate([encoding, observations["proprio"]], axis=-1)

        if self.stop_gradient:
            encoding = jax.lax.stop_gradient(encoding)

        return encoding


class LCEncodingWrapperM(nn.Module):
    """
    Encodes two camera views (main and wrist) and language embeddings into a single flat encoding.

    Each camera view is passed through its own encoder, both conditioned on the language embedding.
    Their encodings are fused (concatenated or averaged) to produce a single representation.

    Args:
        encoder_main: Encoder network for the main/front camera view.
        encoder_wrist: Encoder network for the wrist camera view.
        fusion: Fusion strategy between encoders ("concat" or "add").
        use_proprio: Whether to concatenate proprioceptive features.
        stop_gradient: Whether to stop the gradient on the final encoding.
    """

    encoder: nn.Module
    cond: str = "concat"          # ["concat", "add", "mean"]
    use_proprio: bool = True
    stop_gradient: bool = False

    def __call__(
        self,
        observations: Dict[str, jnp.ndarray], train:bool,
    ) -> jnp.ndarray:
        if (
            isinstance(observations, flax.core.FrozenDict)
            or isinstance(observations, dict)
            and ("image" in observations or "encoding" in observations)
        ):
            if "encoding" in observations:
                return observations["encoding"]
        #     obs = observations["image"]
        #     if self.enable_stacking:
        #         # Combine stacking and channels into a single dimension
        #         if len(obs.shape) == 4:
        #             obs = rearrange(obs, "T H W C -> H W (T C)")
        #         if len(obs.shape) == 5:
        #             obs = rearrange(obs, "B T H W C -> B H W (T C)")
        # else:
        #     obs = observations
        img_main = observations["image"]          
        language = observations["language"]             

        if len(img_main.shape) == 5:
            batch_size, obs_horizon = img_main.shape[:2]
            img_main = rearrange(img_main, "B T H W C -> (B T) H W C")
            language = rearrange(language, "B T F -> (B T) F")
        else:
            batch_size, obs_horizon = img_main.shape[0], 1
        if self.cond == "concat":
            enc = self.encoder(img_main, train=train)
        else:
            enc = self.encoder(img_main, cond_var=language, train=train)

        if obs_horizon > 1:
            enc = rearrange(enc, "(B T) F -> B T F", B=batch_size, T=obs_horizon)
        else:
            enc = enc[:, None, :]  # (B, F) → (B, 1, F)

        if self.cond =="concat":
            if obs_horizon > 1:
                language = rearrange(language, "(B T) F -> B T F", B=batch_size, T=obs_horizon)
            else:
                language = language[:, None, :]
            enc = jnp.concatenate([enc, language], axis=-1)
        # jax.debug.print("fused shape (before use_proprio): {x}", x=fused.shape)
        # jax.debug.print("proprio shape (before use_proprio): {x}", x=observations['proprio'].shape)
        if self.use_proprio:
            proprio = observations["proprio"]
            # if proprio.ndim == 3:  # (B, T, D)
            #     proprio = rearrange(proprio, "B T D -> B (T D)")
            # elif proprio.ndim == 2:  # (B, D)
            #     # already flat, do nothing
            #     pass
            # else:
            #     raise ValueError(f"Unexpected proprio shape: {proprio.shape}")
            
            fused = jnp.concatenate([enc, proprio], axis=-1)

        if self.stop_gradient:
            fused = jax.lax.stop_gradient(fused)
        # jax.debug.print("final fused shape (before stop_gradient): {x}", x=fused.shape)
        # if fused.ndim == 2:
        #     # (B, F) -> (B, 1, F)
        #     fused = fused[:, None, :]

        return fused

class MultiViewLCEncodingWrapper(nn.Module):
    """
    Encodes two camera views (main and wrist) and language embeddings into a single flat encoding.

    Each camera view is passed through its own encoder, both conditioned on the language embedding.
    Their encodings are fused (concatenated or averaged) to produce a single representation.

    Args:
        encoder_main: Encoder network for the main/front camera view.
        encoder_wrist: Encoder network for the wrist camera view.
        fusion: Fusion strategy between encoders ("concat" or "add").
        use_proprio: Whether to concatenate proprioceptive features.
        stop_gradient: Whether to stop the gradient on the final encoding.
    """

    encoder_main: nn.Module
    encoder_wrist: nn.Module
    fusion: str = "concat"          # ["concat", "add", "mean"]
    use_proprio: bool = True
    stop_gradient: bool = False

    def __call__(
        self,
        observations: Dict[str, jnp.ndarray], train:bool,
    ) -> jnp.ndarray:
        img_main = observations["image"]          
        img_wrist = observations["wrist_image"]
        language = observations["language"]             

        if len(img_main.shape) == 5:
            batch_size, obs_horizon = img_main.shape[:2]
            img_main = rearrange(img_main, "B T H W C -> (B T) H W C")
            img_wrist = rearrange(img_wrist, "B T H W C -> (B T) H W C")
        else:
            batch_size, obs_horizon = img_main.shape[0], 1


        enc_main = self.encoder_main(img_main, cond_var=language, train=train)
        enc_wrist = self.encoder_wrist(img_wrist, cond_var=language, train=train)

        if self.fusion == "concat":
            fused = jnp.concatenate([enc_main, enc_wrist], axis=-1)
        elif self.fusion in ["add", "sum"]:
            fused = enc_main + enc_wrist
        elif self.fusion == "mean":
            fused = 0.5 * (enc_main + enc_wrist)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion}")

        if obs_horizon > 1:
            fused = rearrange(fused, "(B T) F -> B T F", B=batch_size, T=obs_horizon)
        if self.use_proprio:
            fused = jnp.concatenate([fused, observations["proprio"]], axis=-1)

        # --- Optionally stop gradient ---
        if self.stop_gradient:
            fused = jax.lax.stop_gradient(fused)

        return fused

class MultiViewSingleLCEncodingWrapper(nn.Module):
    """
    Encodes two camera views (main and wrist) and language embeddings into a single flat encoding.

    Each camera view is passed through its own encoder, both conditioned on the language embedding.
    Their encodings are fused (concatenated or averaged) to produce a single representation.

    Args:
        encoder_main: Encoder network for the main/front camera view.
        encoder_wrist: Encoder network for the wrist camera view.
        fusion: Fusion strategy between encoders ("concat" or "add").
        use_proprio: Whether to concatenate proprioceptive features.
        stop_gradient: Whether to stop the gradient on the final encoding.
    """

    encoder: nn.Module
    fusion: str = "concat"          # ["concat", "add", "mean"]
    use_proprio: bool = True
    stop_gradient: bool = False

    def __call__(
        self,
        observations: Dict[str, jnp.ndarray], train:bool,
    ) -> jnp.ndarray:
        if (
            isinstance(observations, flax.core.FrozenDict)
            or isinstance(observations, dict)
            and ("image" in observations or "encoding" in observations)
        ):
            if "encoding" in observations:
                return observations["encoding"]
        #     obs = observations["image"]
        #     if self.enable_stacking:
        #         # Combine stacking and channels into a single dimension
        #         if len(obs.shape) == 4:
        #             obs = rearrange(obs, "T H W C -> H W (T C)")
        #         if len(obs.shape) == 5:
        #             obs = rearrange(obs, "B T H W C -> B H W (T C)")
        # else:
        #     obs = observations
        img_main = observations["image"]          
        img_wrist = observations["wrist_image"]
        language = observations["language"]             

        if len(img_main.shape) == 5:
            batch_size, obs_horizon = img_main.shape[:2]
            img_main = rearrange(img_main, "B T H W C -> (B T) H W C")
            img_wrist = rearrange(img_wrist, "B T H W C -> (B T) H W C")
            language = rearrange(language, "B T F -> (B T) F")
        else:
            batch_size, obs_horizon = img_main.shape[0], 1

        enc_main = self.encoder(img_main, cond_var=language, train=train)
        enc_wrist = self.encoder(img_wrist, cond_var=language, train=train)

        if self.fusion == "concat":
            fused = jnp.concatenate([enc_main, enc_wrist], axis=-1)
        elif self.fusion in ["add", "sum"]:
            fused = enc_main + enc_wrist
        elif self.fusion == "mean":
            fused = 0.5 * (enc_main + enc_wrist)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion}")

        if obs_horizon > 1:
            fused = rearrange(fused, "(B T) F -> B T F", B=batch_size, T=obs_horizon)
        else:
            fused = fused[:, None, :]  # (B, F) → (B, 1, F)
        # jax.debug.print("fused shape (before use_proprio): {x}", x=fused.shape)
        # jax.debug.print("proprio shape (before use_proprio): {x}", x=observations['proprio'].shape)
        if self.use_proprio:
            proprio = observations["proprio"]
            # if proprio.ndim == 3:  # (B, T, D)
            #     proprio = rearrange(proprio, "B T D -> B (T D)")
            # elif proprio.ndim == 2:  # (B, D)
            #     # already flat, do nothing
            #     pass
            # else:
            #     raise ValueError(f"Unexpected proprio shape: {proprio.shape}")
            
            fused = jnp.concatenate([fused, proprio], axis=-1)

        if self.stop_gradient:
            fused = jax.lax.stop_gradient(fused)
        # jax.debug.print("final fused shape (before stop_gradient): {x}", x=fused.shape)
        # if fused.ndim == 2:
        #     # (B, F) -> (B, 1, F)
        #     fused = fused[:, None, :]

        return fused