import type { VizFrame } from '../../api/client'

/**
 * Module-level animation state read by Three.js components in useFrame().
 * Bypasses React's render cycle so the 3D scene updates at native framerate.
 *
 * `index` = discrete frame for data readouts (side panel, force arrows).
 * `frac`  = 0..1 interpolation toward the next frame for smooth XY/heading.
 *
 * Written by: PlaybackLoop.useFrame (during playback) and playbackStore.setFrame
 * Read by: WireframeCar, ForceArrows, CameraController (all in their useFrame hooks)
 */
export const animState = {
  frames: [] as VizFrame[],
  index: 0,
  /** Fraction between frames[index] and frames[index+1] for interpolation */
  frac: 0,
  get current(): VizFrame | null {
    return this.frames[this.index] ?? null
  },
}

/** Called by playbackStore.setFrame so scrubber/buttons sync immediately. */
export function syncAnimIndex(i: number) {
  animState.index = i
  animState.frac = 0
}

/** Shortest-path angle lerp (handles wrapping around +/-PI). */
export function lerpAngle(a: number, b: number, t: number): number {
  let diff = b - a
  // Wrap to [-PI, PI]
  while (diff > Math.PI) diff -= 2 * Math.PI
  while (diff < -Math.PI) diff += 2 * Math.PI
  return a + diff * t
}
