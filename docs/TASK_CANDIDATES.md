# Task candidates

## Implemented (Phase 5B)

| Task | Idea | Success metric |
|---|---|---|
| `wall_follow` | Right-wall standoff while moving down corridor | Standoff + forward progress; episode ends at corridor exit |
| `go_to_goal` | Point navigation in open arena | Reach goal within 0.25 m (held 20 control ticks) |
| `docking` | Parallel-park into a side bay | Pose in bay + heading aligned to dock yaw |
| `maze` | Navigate obstacle corridor maze | Reach exit region without lidar collision |
| `figure8_tracking` | Smooth lemniscate path tracking | Complete ≥1 loop with low crosstrack / heading error |

Robot: **`diffdrive_lidar`** for all of the above (no new URDF). Worlds differ per task (MuJoCo scene files + PyBullet box layouts).

## Not implementing (explicitly deferred)

- Multi-waypoint patrol
- Uphill climb / terrain grade

## Further ideas (not built)

1. **Dynamic obstacle dodge** — moving box crossing the path; success = reach goal without contact.
2. **Tight U-turn hallway** — minimum-radius turn under lidar; success = reverse heading without crash.
3. **Payload drag** — towed mass changes dynamics; success = goal with bounded slip.
4. **Blind corner peek** — approach T-junction and stop with front clearance band.
5. **Lidar-denied stretch** — intermittent ray dropouts; robustness metric.
6. **Multi-robot yield** — second kinematic agent; success = pass without collision (harder orchestration).
7. **Slope approach** — mild ramp (needs new scene + possibly different robot contact).
8. **Reverse docking** — back into bay (harder than forward parallel park).
