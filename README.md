# MATH 550: Numerical PDEs - Homework 0

## Project Structure

```
MATH550-HW0/
├── README.md
├── requirements.txt
├── code/
│   ├── SecondOrderFiniteDifferenceMethod.py
│   ├── SecondOrderSolveODE.py
│   ├── PoissonEquationSolve.py
│   └── PoissonNeumannPinned.py
└── writeup/
    ├── MATH550_HW0.pdf
    ├── MATH550_HW0.tex
    └── fig/
        ├── Problem1Part1.png
        ├── Problem1Part2.png
        ├── Problem2Part1.png
        ├── Problem2Part2.png
        ├── Problem3Part1.png
        ├── Problem3Part2.png
        ├── Problem4Part1.png
        └── Problem4Part2.png
```

## Files

| Path | Description |
|------|-------------|
| `writeup/MATH550_HW0.tex` | LaTeX source for the write-up |
| `writeup/MATH550_HW0.pdf` | Compiled homework submission |
| `code/SecondOrderFiniteDifferenceMethod.py` | Problems 1–2: first derivative (centered + one-sided) and periodic second derivative of $e^{sin x}$, with convergence plots |
| `code/SecondOrderSolveODE.py` | Problem 3: solves u'' + sin(x)u' + u = f(x) with Dirichlet BCs |
| `code/PoissonEquationSolve.py` | Problem 4.1: 2D Poisson equation with Dirichlet BCs |
| `writeup/fig/Problem1Part1.png` | Exact vs. numerical first derivative |
| `writeup/fig/Problem1Part2.png` | Convergence of the first derivative |
| `writeup/fig/Problem2Part1.png` | Exact vs. numerical second derivative |
| `writeup/fig/Problem2Part2.png` | Convergence of the second derivative |
| `writeup/fig/Problem3Part1.png` | Exact vs. numerical to the ODE |
| `writeup/fig/Problem3Part2.png` | Convergence of the ODE |
| `writeup/fig/Problem4Part1.png` | Numerical Solution to Poisson Equation |
| `writeup/fig/Problem4Part2.png` | Convergence of the 2D Poisson Numerical Solution |

## Running the Code

```bash 
pip install numpy scipy matplotlib
python code/SecondOrderFiniteDifferenceMethod.py
python code/SecondOrderSolveODE.py
python code/PoissonEquationSolve.py
```