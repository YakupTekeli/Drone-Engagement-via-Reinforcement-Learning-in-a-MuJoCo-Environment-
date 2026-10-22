#!/usr/bin/env python3
"""Quadrotor MJCF modelinin temel yapısını yazdır."""

from __future__ import annotations

import mujoco

from ebabil.models import DEFAULT_QUADROTOR_XML, load_quadrotor

ROTOR_SITES = (
    "rotor_front_left",
    "rotor_front_right",
    "rotor_rear_right",
    "rotor_rear_left",
)


def main() -> None:
    model = load_quadrotor()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    root_body_id = model.body("quadrotor").id
    print(f"model: {model.names.decode().split(chr(0))[0] or 'ebabil_quadrotor'}")
    print(f"source: {DEFAULT_QUADROTOR_XML}")
    print(f"timestep_s: {model.opt.timestep:.6f}")
    print(f"mass_kg: {model.body_mass[root_body_id]:.6f}")
    print(f"inertia_kg_m2: {model.body_inertia[root_body_id].tolist()}")
    print(f"nq/nv/nu/nsensor: {model.nq}/{model.nv}/{model.nu}/{model.nsensor}")
    print("rotor_sites_body_m:")
    for site_name in ROTOR_SITES:
        site_id = model.site(site_name).id
        print(f"  {site_name}: {model.site_pos[site_id].tolist()}")


if __name__ == "__main__":
    main()
