from Matrix4Patent import Matrix4Citation
mc = Matrix4Citation()
mode = input("Which matrix? (A=1, B=2): ")
p = input("Which period ? (1, 2, 3, loop=0): ")
if p == '0':
    for p_idx in range(1,4):
        if mode == '1':
            mc.set_matrix_A(p_idx)
        else:
            mc.set_matrix_B(p_idx)
else:
    if mode == '1':
        mc.set_matrix_A(int(p))
    else:
        mc.set_matrix_B(int(p))
