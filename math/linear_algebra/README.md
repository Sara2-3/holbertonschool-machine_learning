# Task 0. Slice Me Up


# Notes
# 1. Concatenation axis=0 (vertical - poshtë):
[[11, 22, 33],
 [44, 55, 66],
 [1,  2,  3],
 [4,  5,  6]]
Rreshtat: 2 + 2 = 4 rreshta
Kolonat: Mbeten 3


# 2. Concatenation axis=1 (horizontal - djathtas):
text
[[11, 22, 33, 1, 2, 3],
 [44, 55, 66, 4, 5, 6]]
Rreshtat: Mbeten 2
Kolonat: 3 + 3 = 6 kolona

axis=0 - bashko poshtë (shto rreshta)

axis=1 - bashko djathtas (shto kolona)

# 14.Write a function def np_matmul(mat1, mat2): that performs matrix multiplication:
Shembull i saktë për Matrix Multiplication:
A (2x3):
text
[[11, 22, 33],
 [44, 55, 66]]
B (3x2):
text
[[1, 2],
 [3, 4], 
 [5, 6]]
Rezultati (2x2):
text
[[11×1 + 22×3 + 33×5,  11×2 + 22×4 + 33×6],
 [44×1 + 55×3 + 66×5,  44×2 + 55×4 + 66×6]]

= [[11+66+165, 22+88+198],
   [44+165+330, 88+220+396]]

= [[242, 308],
   [539, 704]]

NDRYSHIMET KRYESORE:
Element-wise	Matrix Multiplication
A * B	A @ B ose np.matmul(A, B)
Dimensionet duhet të jenë të njëjta	Kolonat e A = Rreshtat e B
Çdo element veç e veç	Kombinim linear i rreshtave/kolonave

rmula e shumëzimit të matricave është:

Formula e Përgjithshme:
Nëse kemi:

Matrica A me dimensione m × n

Matrica B me dimensione n × p

Atëherë C = A × B do të ketë dimensione m × p dhe:

text

