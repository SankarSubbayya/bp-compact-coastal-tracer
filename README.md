# Bound-preserving compact tracer transport for coastal and ocean models

Reproducibility package for:

> S. Sankaranarayanan, "Bound-preserving high-resolution compact tracer transport
> for coastal and ocean models, with application to a calibrated mixed-tide model
> of San Francisco Bay," submitted to *Ocean Engineering* (2026).

All results in the paper regenerate from the scripts in this repository.
Input data (NOAA Coastal Relief Model bathymetry; NOAA Golden Gate tidal
harmonic constants, station 9414290) are fetched from public NOAA services by
the included scripts; no data files are redistributed here.

## Requirements

Python 3 with `numpy`, `scipy`, `matplotlib` (`netcdf4` for the bathymetry
fetch; `sympy` for the symbolic verifications). All commands below use
[`uv`](https://docs.astral.sh/uv/); plain `pip` works equally.

## Reproduce the deformational-flow benchmark (Section 6)

```
uv run --with numpy --with scipy python3 gen_deepen.py     # slotted cylinder, 3 resolutions, 5 schemes
uv run --with numpy --with scipy python3 gen_gauss.py      # smooth-field (Gaussian) test
uv run --with numpy --with scipy python3 gen_global.py     # global vs DMP limiter bounds
uv run --with numpy --with scipy python3 run_cfl.py        # time-step sensitivity
uv run --with numpy --with scipy --with matplotlib python3 gen_plots.py   # benchmark figures
uv run --with sympy python3 sympy_proofs.py                # symbolic order/wavenumber checks
```

## Reproduce the San Francisco Bay application (Section 6.5)

```
uv run --with netcdf4 --with numpy python3 fetch_crm.py           # NOAA CRM bathymetry subset
uv run --with numpy --with scipy python3 build_sfbay_fine.py      # 200 m grid
uv run --with numpy --with scipy python3 gen_currents_mixed.py    # calibrated mixed-tide currents
uv run --with numpy --with scipy --with matplotlib python3 sfbay_tides_fig.py
uv run --with numpy --with scipy --with matplotlib python3 sfbay_currents_fig.py
uv run --with numpy --with scipy --with matplotlib python3 sfbay_domain.py
uv run --with numpy --with scipy --with matplotlib python3 sfbay_transport.py
uv run --with numpy --with scipy python3 run_sfbay_tvd.py         # TVD comparator (advective form)
uv run --with numpy --with scipy python3 run_sfbay_bpglobal.py    # global-bounds limiter variant
```

Hydrodynamic calibration details and acceptance targets are documented in
`CALIBRATION.md`. Reference outputs for every table in the paper are archived
under `results/`.

## Layout

- `advdiff.py`, `sharp.py`, `compact_stable.py` — shared numerics modules
- `gen_*.py`, `run_*.py` — experiment drivers (see above)
- `sfbay_fine/`, `figures/` — generated grids/fields and figures
- `results/` — archived result JSONs used by the paper's tables

## License

MIT (see LICENSE).
