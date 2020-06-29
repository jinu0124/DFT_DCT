import os
import numpy as np
import math
from PIL import Image
from matplotlib import pyplot as plt
import random

# 2015253039 권진우 영상처리 HW5

COEFF = 0.707106781

class performer:
    def __init__(self, root_dir):
        super().__init__()

        print("2015253039 권진우 영상처리 HW5")

        self.image_select = ''  # Image Select (BOAT512 or Lena_512x512)
        boat512 = self.initialize(root_dir) # 초기화 수행 -> 이미지 선택, 받아오기
        self.root_dir = root_dir # Directory PATH
        self.dst_re_im = np.zeros((len(boat512), len(boat512[0])), dtype=complex)  # Using in Inverse FFT, frequency domain static values
        self.after_dct_image = np.zeros((len(boat512), len(boat512[0])), dtype=float) # Using in Inverse DCT, frequency domain static values
        self.after_dct_image_block_size = 0

        cont = 1  # return 값이 1 이면 계속해서 프로그램을 종료시키지 않고 반복 수행
        while cont is 1:
            cont = self.control(boat512)
            # cont == 2 일 때 <- 이미지 다시 선택하기 기능
            if cont is 2:
                boat512 = self.initialize(root_dir)
                # 다룰 이미지가 바뀌었으므로 FFT 후의 복소수 배열 값을 초기화 할 필요가 있다.
                self.dst_re_im = np.zeros((len(boat512), len(boat512[0])), dtype=complex)
                cont = 1

        exit() # 프로그램 종료

    # initialize : .Raw Image 파일 형식에 맞게 Gray Scale, [512, 512] Import, Image Select (BOAT512 / LENA)
    def initialize(self, root_dir):
        def get_boat_raw(root_dir):
            a = input("1. BOAT512.raw\n2. Lena.raw")
            boat_image_dir = ''
            image_sel = ''

            if int(a) is 1:
                boat_image_dir = os.path.join(root_dir, 'BOAT512.raw')
                image_sel = 'BOAT512'
            elif int(a) is 2:
                boat_image_dir = os.path.join(root_dir, 'lena_raw_512x512.raw')
                image_sel = 'Lena'
            else:
                exit()
            scene_infile = open(boat_image_dir, 'rb')
            scene_image_array = np.fromfile(scene_infile, dtype=np.uint8, count=512 * 512)
            boat_image = Image.frombuffer("L", [512, 512], scene_image_array, 'raw', 'L', 0, 1)  # L : 8bit gray Scale
            boat_image = np.array(boat_image, dtype=float)  # numpy 2차 배열에 담기, #int64로 Casting하여 MSE계산 시 Overflow 방지
            plt.imsave(image_sel + '.bmp', boat_image, cmap='gray')

            return boat_image, image_sel

        # Test 용
        def get_boat_200(root_dir):
            boat_image_dir = os.path.join(root_dir, 'Resized_boat_image.bmp')
            boat_image = Image.open(boat_image_dir).convert('L') # Gray Scale로 받기
            boat_image = np.array(boat_image)
            print(len(boat_image))
            print(len(boat_image[0]))

            plt.imshow(boat_image, cmap='gray')
            plt.show()

            return boat_image

        # get BOAT512 image into Numpy 2-Dimension Array # 과제용 512x512
        boat512, image_sel = get_boat_raw(root_dir)
        self.image_select = image_sel

        # testing 200x200.bmp image # 테스트용 200x200 <- 빠른 Testing을 위함
        # boat512 = get_boat_200(root_dir)

        return boat512

    def butterflies(self, numpoints, logN, dir, f):
        N = 1

        for k in range(logN):
            w = complex(0.0, 0.0)
            wp = complex(0.0, 0.0)
            temp = complex(0.0, 0.0)

            half_N = N
            N <<= 1
            angle = -2.0 * math.pi / N * dir
            wtemp = math.sin(0.5 * angle)
            wp = complex((-2.0 * wtemp * wtemp), (math.sin(angle)))
            w = complex(1.0, 0.0)
            for offset in range(0, half_N):
                for i in range(offset, numpoints, N):
                    j = i + half_N
                    temp = complex(((w.real * f[j].real) - (w.imag * f[j].imag)), 0.0)
                    temp = complex(temp.real,
                                   ((w.imag * f[j].real) + (w.real * f[j].imag)))
                    f[j] = complex((f[i].real - temp.real), f[j].imag)
                    f[i] = complex(f[i].real + temp.real, f[i].imag)
                    f[j] = complex(f[j].real, (f[i].imag - temp.imag))
                    f[i] = complex(f[i].real, (f[i].imag + temp.imag)) # 순서대로 계산 해주어야한다 *** 중요! <- 한번에 f[j], f[i] = x, x 하면 안됨.
                wtemp = w.real
                w = complex((wtemp * wp.real - w.imag * wp.imag + w.real), (w.imag * wp.real + wtemp * wp.imag + w.imag))

        if dir is -1:
            for i in range(numpoints):
                f[i] = complex(f[i].real/numpoints, f[i].imag)
                f[i] = complex(f[i].real, f[i].imag/numpoints)
                # == f[i] = complex((f[i].real/numpoints), (f[i].imag/numpoints))

    def scramble(self, numpoints, f):
        j = 0
        for i in range(numpoints):
            if i > j:
                f[i], f[j] = complex(f[j].real, f[j].imag), complex(f[i].real, f[i].imag)
                # f.real[j], f.real[i] = f.real[i], f.real[j]  # swap
                # f.imag[j], f.imag[i] = f.imag[i], f.imag[j]  # swap
            m = numpoints >> 1
            while (j >= m) & (m >= 2):
                j -= m
                m = m >> 1
            j += m

    # FFT 수행 : f : self.row_data/self.col_data, logN : M, numpoints : 512, dir : Forward/Inverse Direction
    def fft(self, f, logN, numpoints, dir):
        self.scramble(numpoints, f)
        self.butterflies(numpoints, logN, dir, f)

    # dft(fft 사용) 수행
    def dft(self, boat512, dir):
        self.row_data = np.zeros((len(boat512)), dtype=complex)
        self.col_data = np.zeros((len(boat512)), dtype=complex)

        self.re = np.zeros((len(boat512) * len(boat512[0])), dtype=float)
        self.im = np.zeros((len(boat512) * len(boat512[0])), dtype=float)

        # direction이 Forward이면 원영상(Time Domain)이 real값만 가지므로 self.re 배열에 boat512[][] 이미지를 1차원으로 정렬하여 담는다.
        # direction이 Inverse이면 DFT후의 Frequency Domain은 Real값과 Imag값을 모두 가지므로 실수, 허수를 모두 1차원으로 정렬하여 담는다.
        if dir is 1:
            for i in range(len(boat512)):
                for j in range(len(boat512[0])):
                    self.re[i * len(boat512) + j] = boat512[i][j]
        if dir is -1:
            for i in range(len(boat512)):
                for j in range(len(boat512[0])):
                    self.re[i * len(boat512) + j] = boat512[i][j].real
                    self.im[i * len(boat512) + j] = boat512[i][j].imag

        rows = len(boat512)
        cols = len(boat512[0])
        M, N = 0, 0
        # dir = 1  # flag 값 : 1=fft, -1=Inverse_fft

        flag = 0
        num = 0
        # compute power of 2s
        for i in range(2):
            num = cols if flag is 0 else rows
            while num >= 2:
                num >>= 1
                if flag is 0:
                    M += 1
                else:
                    N += 1
            flag = 1
        print('M :', M, ', N :', N)

        # 2 Dimension Compute를 해도 되지만 계산 복잡도가 더 낮은 1차원 DCT를 행/열 방향으로 각각 계산한다.
        # 행 방향
        for y in range(rows):
            if y % 50 is 0:
                print('processing', y + 1, 'row')
            # 행 방향이므로 index는 y x cols(512)로 행(줄)단위로 넘어간다.
            index = y * cols
            # self.row_data[]에 1줄(행)을 담는다. # Python에서는 complex() 복소수를 complex(실수, 허수)로 표현하며 실수, 허수를 따로 할당은 불가하다.
            # Ex. real_num = complex(self.re[])  <- (X) 불가, real_num = complex(self.re[], real_num[].imag) <- (0) 가능
            for x in range(cols):
                self.row_data[x] = complex(self.re[index], self.im[index])
                index += 1

            # FFT 수행 (1행 씩)
            self.fft(self.row_data, M, cols, dir)
            index = y * cols
            # Real, imag 값을 각각 분리하여서 self.re, self.im 배열에 담는다.
            for x in range(cols):
                self.re[index] = self.row_data[x].real
                self.im[index] = self.row_data[x].imag
                index += 1

        # 열 방향
        for x in range(cols):
            if x % 50 is 0:
                print('processing', x + 1, 'cols')
            index = x
            for y in range(rows):
                self.col_data[y] = complex(self.re[index], self.im[index])
                index += cols

            self.fft(self.col_data, N, rows, dir) # fft 수행
            index = x
            for y in range(rows):
                self.re[index] = self.col_data[y].real
                self.im[index] = self.col_data[y].imag
                index += cols

        dst = np.zeros((len(boat512), len(boat512[0])), dtype=float)  # == [[0.0 for i in range(len(boat512))] for j in range(len(boat512))]

        # Inverse DFT 후 Output으로 나온 real, imag(허수) 값을 sqrt(real**2 + imag**2)를 통해서 출력해주고 저장한다. + dst 반환
        if dir is -1:
            for i in range(len(boat512)):
                for j in range(len(boat512[0])):
                    dst[i][j] = math.sqrt((self.re[i * len(boat512) + j])**2 + (self.im[i * len(boat512) + j])**2)
            plt.imsave(self.root_dir + '/after_Inverse_dft_' + self.image_select + '.bmp', dst, cmap='gray')

        # Forward DFT 후 Output으로 나온 Real, imag 값을 log함수를 사용하여 출력하고 저장한다.
        if dir is 1:
            for i in range(len(boat512)):
                for j in range(len(boat512[0])):
                    # plot를 통해 출력을 보여주기 위해서 D(u,v) = c * log|1+|H(u,v)||
                    dst[i][j] = N * math.log(abs(1 + abs(complex(self.re[i * len(boat512) + j], self.im[i * len(boat512) + j]))))
                    # self.dst_re_im에 log가 적용되지 않은 real, imag 복소수를 가지는 Frequency Domain을 저장해둔다. -> Inverse에 사용된다.
                    self.dst_re_im[i][j] = complex(self.re[i * len(boat512) + j], self.im[i * len(boat512) + j])
            plt.imsave(self.root_dir + '/after_dft_log_option_' + self.image_select + '.bmp', dst, cmap='gray')

        plt.imshow(dst, cmap='gray')
        plt.show()

        # Inverse 일 때 inverse_dst를 return
        if dir is -1:
            return dst
        return 0

    # MEAN Square Error를 계산 : DST_image와 SRC_image를 받아서 MSE를 계산 후 Return 시킨다.
    def MSE(self, dst_image, origin_image):
        Accum = 0.0
        for r in range(len(dst_image)):
            for c in range(len(dst_image[r])):
                Accum += ((dst_image[r][c] - origin_image[r][c]) ** 2).astype(float)  # 차이값^2을 누적하여서
        mse = Accum / (len(dst_image) * len(dst_image[0]))  # 전체 픽셀 수로 나누기
        return mse

    # formula to dft
    def formula(self, boat512, flag):
        # boat512 : boat Image Numpy 2-dimension Array
        # flag : direction [1:Forward DFT, -1:Backward(Inverse) DFT]
        if flag is 1:
            _ = self.dft(boat512, flag)
            return 0
        elif flag is -1:
            inverse_dft = self.dft(boat512, flag)
            return inverse_dft

    # Ex) N=8이라면 512/8 = x,y 각 64개의 block 씩 DCT 수행
    # Y축 방향 먼저 (상 - > 하 방향)
    # perform 1-D DCT (인자 순서 : 블럭개수, 결과로 출력될 배열, Block Size, Input Image, Forward/Inverse Direction)
    def dct_y(self, block_num, dst, N, boat, dir):
        accum = 0.0
        k1 = 0.0
        for y in range(block_num):
            if y % 20 is 0 and (N is 8 or N is 4):
                print(y*N+1, "rows")
            elif N >= 32:
                print(y*N+1, "rows")
            for x in range(block_num):
                for yy in range(N):
                    if (block_num is 1 or block_num is 2) and yy % 10 == 0:
                        print(yy+1, "rows")
                    for xx in range(N):
                        # Forward와 Inverse DCT에 따라서 수행되는 절차가 다르다.
                        # Forward는 1 줄에 대한 값을 누적하여 Scale을 곱하며, Backward는 1 줄에 대한 계산 중에 매번 Scale을 곱하고 누적된 값을 할당할때는 적용하지 않는다.
                        if dir is 1:
                            # AC, DC에 따라서 k(u) 값을 알맞게 적용시킨다.
                            if xx is 0:
                                k1 = 1 / math.sqrt(N)
                            else:
                                k1 = math.sqrt(2/N)
                            for n in range(N):
                                accum += (math.cos(((2 * n + 1) * xx * math.pi) / (2 * N)) * boat[x * N + n][
                                    y * N + yy])
                            dst[x * N + xx][y * N + yy] = accum * k1
                        else:
                            for n in range(N):
                                if n is 0:
                                    k1 = 1 / math.sqrt(N)
                                else:
                                    k1 = math.sqrt(2 / N)
                                accum += (k1 * math.cos(((2 * xx + 1) * n * math.pi) / (2 * N)) * boat[x * N + n][
                                    y * N + yy])
                            dst[x * N + xx][y * N + yy] = accum
                        accum = 0.0
        return dst

    # perform 1-D DCT
    def dct_x(self, block_num, final_dst, N, dst, dir):
        k2 = 0.0
        # X축 방향 (--> 방향)
        accum = 0.0
        for x in range(block_num):
            if x % 20 is 0 and (N is 8 or N is 4):
                print(x * N + 1, "cols")
            elif N >= 32:
                print(x * N + 1, "cols")
            for y in range(block_num):
                for xx in range(N):
                    if (block_num is 1 or block_num is 2) and xx % 10 == 0:
                        print(xx + 1, "cols")
                    for yy in range(N):
                        # Forward와 Inverse DCT에 따라서 수행되는 절차가 다르다.
                        # Forward는 1 줄에 대한 값을 누적하여 Scale을 곱하며, Backward는 1 줄에 대한 계산 중에 매번 Scale을 곱하고 누적된 값을 할당할때는 적용하지 않는다.
                        if dir is 1:
                            if yy is 0:
                                k2 = 1 / math.sqrt(N)
                            else:
                                k2 = math.sqrt(2 / N)
                            for n in range(N):
                                accum += (math.cos(((2 * n + 1) * yy * math.pi) / (2 * N)) * dst[x * N + xx][y * N + n])
                            final_dst[x * N + xx][y * N + yy] = accum * k2
                        else:
                            for n in range(N):
                                if n is 0:
                                    k2 = 1 / math.sqrt(N)
                                else:
                                    k2 = math.sqrt(2 / N)
                                accum += (k2 * math.cos(((2 * yy + 1) * n * math.pi) / (2 * N)) * dst[x * N + xx][
                                    y * N + n])
                            final_dst[x * N + xx][y * N + yy] = accum
                        accum = 0.0
        return final_dst

    # perform 2D dimension DCT
    def dct_perform_2d(self, block_num, final_dst, N, src, dir):
        accum = 0.0
        scale = 2.0/math.sqrt(N*N)
        for x in range(block_num):
            for y in range(block_num):
                if (x * block_num + y)%400 == 0 and (N is 8 or N is 4):
                    print(x * block_num + y + 1, 'blocks')
                elif N >= 16:
                    print(x * block_num + y + 1, 'blocks')
                for xx in range(N):
                    for yy in range(N):
                        accum = 0
                        Cu = COEFF if xx is 0 else 1.0
                        Cv = COEFF if yy is 0 else 1.0
                        for u in range(N):
                            for v in range(N):
                                accum += (math.cos((math.pi * xx * (2*u+1))/(2 * N)) * math.cos(
                                    (math.pi * yy * (2*v+1))/(2 * N)) * src[x * N + u][y * N + v])
                        accum *= scale
                        accum *= Cu
                        accum *= Cv
                        final_dst[x * N + xx][y * N + yy] = accum
        return final_dst

    # perform 2D dimension Inverse-DCT
    def inverse_dct_perform_2d(self, block_num, final_dst, N, dst, dir):
        accum = 0.0
        scale = 2.0 / math.sqrt(N * N)
        for x in range(block_num):
            for y in range(block_num):
                if (x * block_num + y) % 400 == 0 and (N is 8 or N is 4):
                    print(x * block_num + y + 1, 'blocks')
                elif N >= 16:
                    print(x * block_num + y + 1, 'blocks')
                for xx in range(N):
                    for yy in range(N):
                        for u in range(N):
                            for v in range(N):
                                Cu = COEFF if u is 0 else 1.0
                                Cv = COEFF if v is 0 else 1.0
                                accum += ((Cu * Cv * math.cos((math.pi * u * (2 * xx + 1)) / (2 * N))) * math.cos(
                                    (math.pi * v * (2 * yy + 1)) / (2 * N)) * dst[x * N + u][y * N + v])
                        final_dst[x * N + xx][y * N + yy] = accum * scale
                        accum = 0
        return final_dst

    # DCT, Inverse-DCT 수행을 전적으로 Control하는 부분
    def dct(self, boat, dir, N):
        # 일단 N = 8
        if dir is 1:
            print("Forward DCT 수행")
        else:
            print("Inverse DCT 수행")

        self.cos_macro = [] # 나중에 cos macro 추가 가능

        block_num = len(boat)/N
        # Block Size가 이미지의 크기와 정확히 맞아 떨어지는지 정합성 검사
        if len(boat) % int(block_num) != 0.0:
            print("Image의 크기가 Block의 Size로 나누어 떨어지지 않습니다.")
            print("block number =", block_num)
            return 10
        block_num = int(block_num)

        dst = np.zeros((len(boat), len(boat[0])), dtype=float) # dst = 1차 DCT 중간 결과
        final_dst = np.zeros((len(boat), len(boat[0])), dtype=float) # final_dst = 1차 DCT 최종 결과

        sel = int(input('선택하시오.\n1. 1-D DCT 사용, 2. 2-D DCT 사용\n'))
        if dir is 1:
            if sel is 1:
                # Y 방향으로 1-D DCT
                dst = self.dct_y(block_num, dst, N, boat, dir)
                print('Y방향 완료')

                # X 방향으로 1-D DCT
                final_dst = self.dct_x(block_num, final_dst, N, dst, dir)
                print('X방향 완료')
            else:
                # Perform 2-D DCT
                final_dst =self.dct_perform_2d(block_num, final_dst, N, boat, dir)
        else:
            if sel is 1:
                # X 방향으로 1-D DCT
                dst = self.dct_x(block_num, dst, N, boat, dir)
                print('X방향 완료')

                # Y 방향으로 1-D DCT
                final_dst = self.dct_y(block_num, final_dst, N, dst, dir)
                print('Y방향 완료')
            else:
                # if dir = -1 : Perform 2-D Inverse DCT
                final_dst = self.inverse_dct_perform_2d(block_num, final_dst, N, boat, dir)

        if dir is 1:
            self.after_dct_image = final_dst
            self.after_dct_image_block_size = N
            plt.imsave(self.root_dir + '/after_DCT_' + self.image_select + str(N) + '.bmp', final_dst, cmap='gray')
        else:
            plt.imsave(self.root_dir + '/after_inverse_DCT_' + self.image_select + str(N) + '.bmp', final_dst, cmap='gray')

        plt.imshow(final_dst, cmap='gray')
        plt.show()
        return final_dst # return dst image

    def inverse_dct(self, boat, N):
        print('DCT 값을 얻기 위해서 Forward DCT를 먼저 수행합니다.')

        val = self.dct(boat, 1, N)
        if val is 10:
            return

        print('Inverse DCT를 수행합니다.')
        inverse_dct_image = self.dct(val, -1, N)

        mse = self.MSE(inverse_dct_image, boat)
        print(str(N) + 'x' + str(N) + 'Block Size MSE :', mse)
        return

    # 저장된 DCT 영상 확인하기
    def confirm_dct(self, boat):
        image_list = os.listdir(self.root_dir)
        bmp_list = []
        for i in image_list:
            ext = os.path.splitext(i)
            if 'bmp' in str(ext[1]):
                bmp_list.append(i)

        # 선택한 이미지에 대한 DCT 된 이미지가 있는지 확인하는 작업
        # Dictionary 형태로 저장하여 Key, Values로 쌍으로 저장한다.
        bmp_list_refined = dict()
        for i in range(1, 513):
            temp = 'after_DCT_' + self.image_select + str(i) + '.bmp'
            if str(temp) in bmp_list:
                bmp_list_refined[str(i)] = str(temp)

        if len(bmp_list_refined) is 0:
            print('DCT 수행된 영상이 없으므로 되돌아 갑니다.')
            return
        #
        # inverse_bmp_list_refined = dict()
        # for i in range(1, 513):
        #     temp = 'after_inverse_DCT_' + self.image_select + str(i) + '.bmp'
        #     if str(temp) in bmp_list:
        #         inverse_bmp_list_refined[str(i)] = str(temp)
        #
        # if len(inverse_bmp_list_refined) is 0:
        #     print('Inverse-DCT 수행된 영상이 없으므로 되돌아 갑니다.')
        #     return
        #
        # for i, e in inverse_bmp_list_refined.items():
        #     # 각 영상을 가지고와서 MSE를 계산하여 출력해준다. 또한 int64 type으로 Casting하여 MSE 계산 시 Overflow를 방지한다.
        #     image = np.array(Image.open(os.path.join(self.root_dir, str(e))).convert('L'), dtype=float)
        #     mse = self.MSE(image, boat)
        #     print(e + '의 MSE :', mse)

        print('다음과 같은 영상을 찾았습니다.\n어떤 영상을 보여줄까요?')
        flag = 1
        order = [0 for i in range(200)] # 최대 200개 까지 저장 가능
        for i, e in bmp_list_refined.items():
            print(flag, '.', e)
            order[flag] = int(i)
            flag += 1

        # 사용자로 부터 Inverse_DCT를 수행할 영상을 선택받는다.
        select = input()
        image_file_name = bmp_list_refined[str(order[int(select)])] # 사용자가 선택한 번호의 Key 값을 받아서 해당 Key의 Value(= Image File 이름)를 사용한다.

        image_path = os.path.join(self.root_dir, image_file_name)
        image = np.array(Image.open(image_path).convert('L')) # Get image in Gray scale
        plt.imshow(image, cmap='gray')
        plt.show()

        return

    # Runtime동안 가장 최근 수행된 DCT 배열이 있다면 전역으로 저장되어있으므로 Forward를 수행하지 않고 즉시 Inverse-DCT를 수행한다.
    def direct_dct(self, boat):
        compare = np.zeros((512,512), dtype=float)
        if (self.after_dct_image[10] is compare[10] and self.after_dct_image[100] is compare[100]) or self.after_dct_image_block_size is 0:
            print("최근 배열에 저장된 DCT 수행된 영상이 없습니다.")
            return
        else:
            print('block size:', self.after_dct_image_block_size, 'x', self.after_dct_image_block_size)
            final_dst = self.dct(self.after_dct_image, -1, self.after_dct_image_block_size)

            mse = self.MSE(final_dst, boat)
            print(str(self.after_dct_image_block_size) + 'x' + str(self.after_dct_image_block_size) + 'Block Size 에서 원 영상과의 MSE :', mse)

    # Control Main
    def control(self, boat):
        print('\nSelect Menu')
        print('1. DFT\n2. Inverse DFT & MSE\n3. DCT\n4. 저장된 DCT이미지 확인하기\n5. Inverse DCT 수행 & MSE check\n'
              '6. 가장 최근 수행된적 있는 DCT에 대한 Inverse-DCT 수행\n7. Image Select\nElse. Exit Program')
        a = str(input())

        if a is '1':
            self.forward_dft(boat)
            return 1
        elif a is '2':
            inverse_dft_img = self.inverse_dft(boat)
            # send  after Inverse DFT image & Origin BOAT512 Image in 2-Dimension array
            mse = self.MSE(inverse_dft_img, boat)
            print("MSE :", mse)
            return 1
        elif a is '3':
            print('block size를 선택해주세요.')
            b = input('0. 4x4\n1. 8x8\n2. 64x64\n3. 128x128\n4. 512x512\n5. 원하는 Block Size 수동 입력하기.\n6. Exit Program')
            if int(b) is 0:
                dst = self.dct(boat, 1, 4)
            elif int(b) is 1:
                dst = self.dct(boat, 1, 8)
            elif int(b) is 2:
                dst = self.dct(boat, 1, 64)
            elif int(b) is 3:
                dst = self.dct(boat, 1, 128)
            elif int(b) is 4:
                dst = self.dct(boat, 1, 512)
            elif int(b) is 5:
                c = input('Block Size를 입력해주세요. 512의 약수가 되어야합니다.')
                dst = self.dct(boat, 1, int(c))
            else:
                return 0
            return 1
        elif a is '4':
            _ = self.confirm_dct(boat)
            return 1
        elif a is '5':
            d = input('0. 4x4\n1. 8x8\n2. 64x64\n3. 128x128\n4. 512x512\n5. 원하는 Block Size 수동 입력하기.\n6. Exit')
            if int(d) is 0:
                dst = self.inverse_dct(boat, 4)
            elif int(d) is 1:
                dst = self.inverse_dct(boat, 8)
            elif int(d) is 2:
                dst = self.inverse_dct(boat, 64)
            elif int(d) is 3:
                dst = self.inverse_dct(boat, 128)
            elif int(d) is 4:
                dst = self.inverse_dct(boat, 512)
            elif int(d) is 5:
                e = input('Block Size를 입력해주세요. 512의 약수가 되어야합니다.')
                dst = self.inverse_dct(boat, int(e))
            else:
                return 0
            return 1
        elif a is '6':
            self.direct_dct(boat)
            return 1
        elif a is '7':
            print('이미지를 다시 선택합니다.')
            return 2
        else:
            return 0

    # Forward FFT
    def forward_dft(self, boat):
        _ = self.formula(boat, 1)

    # Inverse FFT
    # Inverse를 수행하기 전에 Forward를 수행하여서 허수를 포함하는 Frequency Domain으로 변환 후 Inverse를 수행합니다.
    # Inverse를 바로 수행하고 싶다면 Frequency Domain 값을 저장해두었다가 다시 불러오는 방향으로 수정가능합니다.
    def inverse_dft(self, boat):
        compare = np.zeros((len(boat), len(boat[0])), dtype=complex)

        # Inverse
        # 이미 self.dst_re_im에 DFT가 수행된 배열 값이 들어있다면 Forward를 수행하지 않고 즉시, Inverse-DFT를 수행한다.
        if self.dst_re_im[10][10].real == compare[10][10].real and self.dst_re_im[10][10].imag == compare[10][10].imag:
            # 먼저 Forward DFT를 수행하여서 허수를 포함하는 Frequency Domain 값을 얻는다.
            print('Frequency Domain 획득을 위해 forward_dft를 먼저 수행합니다.')
            _ = self.formula(boat, 1)
            print('Forward dft가 완료되었습니다.\nInverse dft를 시작합니다.')
        inverse_dft_img = self.formula(self.dst_re_im, -1)
        return inverse_dft_img
