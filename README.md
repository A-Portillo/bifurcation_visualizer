# Bifurcation Lab

A Streamlit explorer for autonomous scalar ODEs dx/dt = f(x, r), using x as the state, t as time, and r as the parameter.

Includes saddle-node, transcritical, supercritical and subcritical pitchfork models, plus custom polynomials, equilibrium stability, phase lines, trajectories, and CSV export.

## Run locally

Use Python 3.11 or 3.12:

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Put app.py, dynamics.py and requirements.txt in a GitHub repository.
2. Create an app in Streamlit Community Cloud and select that repository and branch.
3. Set the main file to app.py and select Python 3.11 or 3.12, then deploy.

No secrets or external services are required.

## Custom expressions

Enter the right-hand side only, for example `r*x - x^3` or `(r-1)*x + x^3 - x^5`. Both ^ and ** mean exponentiation. Multiplication must be explicit. Division is allowed only by nonzero numeric constants; powers must be nonnegative integers. Only x and r are allowed: time-dependent equations do not have this autonomous equilibrium diagram. Total degree is limited to 20 to keep the public app responsive. Input is parsed through a restricted arithmetic AST, without evaluating Python input.

The diagram uses sampled numerical polynomial roots. High degree, near-multiple roots and ill-conditioned coefficients can cause missed or inaccurate branches. Candidate critical parameters from discriminants and leading coefficients are included for total degree up to 8. These candidates do not automatically imply bifurcations. Trajectories terminate at the displayed state bounds.

Run mathematical and app smoke checks with `python -m unittest discover -s tests -v`.

Inspired by https://demonstrations.wolfram.com/BifurcationsInFirstOrderODEs/ . The implementation uses the standard normal forms; it does not reproduce the original demonstration's code or assets.
