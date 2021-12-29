import math
import numpy as np
import PIL
from PIL import Image
from collections import deque
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

def xor(s1, s2):
    xor_s = ''
    for i in range(len(s1)):
        if s1[i] == s2[i]:
            xor_s += '0'
        else:
            xor_s += '1'
            
    return xor_s
            
def left_shift(s, n):
    res = ''
    de = deque(list(s))
    de.rotate(n)
    for i in de:
        res += str(i)
        
    return res
        
def right_shift(s, n):
    res = ''
    de = deque(list(s))
    de.rotate(-n)
    for i in de:
        res += str(i)
        
    return res

def decimalToBinary(n):
    
    bin_n = bin(n).replace("0b", "")
    
    if len(bin_n) < 8:
        temp_0 = ""
        for i in range(8 - len(bin_n)):
            temp_0 += '0'
            
        bin_n = temp_0 + bin_n
    
    return bin_n

def sum_pixels(img_arr):
    sum_pixel = 0
    
    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):
            for k in range(img_arr.shape[2]):
                sum_pixel += img_arr[i,j,k]  
                
    meta_key_bin = decimalToBinary(sum_pixel)
    
    if len(meta_key_bin) < 128:
        less_bits = 128 - len(meta_key_bin)
        t = ''
        for i in range(less_bits):
            if i % 2 == 0:
                t += '1'
            else:
                t += '0'
        meta_key_bin = meta_key_bin + t
        
    return meta_key_bin

def get_meta_key(og_key, meta_key_bin):
    
    xor_s = xor(meta_key_bin, og_key)
    meta_key = int(xor_s, 2) / (math.pow(2, 128))
        
    return round(meta_key, 19)


def keygen(x, r, size):
    
    key = []
    for i in range(size):
        x = x*r*(1-x)
        key.append(int((x * math.pow(10,16))% 256))
        
    key = np.array(key)
    key = np.reshape(key, (img_arr.shape[0], img_arr.shape[1], img_arr.shape[2]))
    
    return key


def Lorenzkey(x0, y0, z0, num_steps):
    
    dt = 0.01
    xs = np.empty(num_steps + 1)
    ys = np.empty(num_steps + 1)
    zs = np.empty(num_steps + 1)
    xs[0], ys[0], zs[0] = x0, y0, z0
    
    s = 10
    r = 28
    b = 8/3
    
    for i in range(num_steps):
        xs[i+1] = (xs[i] + (s * ys[i] -xs[i] * dt)) % 256
        ys[i+1] = (ys[i] + ((xs[i] * (r - zs[i]) - ys[i]) * dt)) % 256
        zs[i+1] = (zs[i] + ((xs[i] * ys[i] - b * zs[i]) * dt)) % 256
        
    return xs, ys, zs


def decToBinary_dims(key):

    first_dim = []
    second_dim = []
    third_dim = []

    for i in range(key.shape[0]):
        
        temp_li1 = []
        temp_li2 = []
        temp_li3 = []

        for j in range(key.shape[1]):
            
            temp_li1.append(decimalToBinary(key[i,j,0]))
            temp_li2.append(decimalToBinary(key[i,j,1]))
            temp_li3.append(decimalToBinary(key[i,j,2]))

        first_dim.append(temp_li1)
        second_dim.append(temp_li2)
        third_dim.append(temp_li3)
    
    return first_dim, second_dim, third_dim


def get_mat(img_arr, encoding_rule1, first_dim, second_dim, third_dim):
    mat = np.empty(img_arr.shape)

    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):

            last2_current_1 = first_dim[i][j][6:]
            first2_next_1 = first_dim[i][(j+1)%img_arr.shape[1]][:2]

            if (encoding_rule1[last2_current_1] == 'A' and encoding_rule1[first2_next_1] == 'T'):
                mat[i][j][0] = 1
            elif (encoding_rule1[last2_current_1] == 'T' and encoding_rule1[first2_next_1] == 'A'):
                mat[i][j][0] = 1
            elif (encoding_rule1[last2_current_1] == 'C' and encoding_rule1[first2_next_1] == 'G'):
                mat[i][j][0] = 1
            elif (encoding_rule1[last2_current_1] == 'G' and encoding_rule1[first2_next_1] == 'C'):
                mat[i][j][0] = 1
            else:
                mat[i][j][0] = 0

            last2_current_2 = second_dim[i][j][6:]
            first2_next_2 = second_dim[i][(j+1)%img_arr.shape[1] ][:2]

            if (encoding_rule1[last2_current_2] == 'A' and encoding_rule1[first2_next_2] == 'T'):
                mat[i][j][1] = 1
            elif (encoding_rule1[last2_current_2] == 'T' and encoding_rule1[first2_next_2] == 'A'):
                mat[i][j][1] = 1
            elif (encoding_rule1[last2_current_2] == 'C' and encoding_rule1[first2_next_2] == 'G'):
                mat[i][j][1] = 1
            elif (encoding_rule1[last2_current_2] == 'G' and encoding_rule1[first2_next_2] == 'C'):
                mat[i][j][1] = 1
            else:
                mat[i][j][1] = 0

            last2_current_3 = third_dim[i][j][6:]
            first2_next_3 = third_dim[i][(j+1)%img_arr.shape[1] ][:2]

            if (encoding_rule1[last2_current_3] == 'A' and encoding_rule1[first2_next_3] == 'T'):
                mat[i][j][2] = 1
            elif (encoding_rule1[last2_current_3] == 'T' and encoding_rule1[first2_next_3] == 'A'):
                mat[i][j][2] = 1
            elif (encoding_rule1[last2_current_3] == 'C' and encoding_rule1[first2_next_3] == 'G'):
                mat[i][j][2] = 1
            elif (encoding_rule1[last2_current_3] == 'G' and encoding_rule1[first2_next_3] == 'C'):
                mat[i][j][2] = 1
            else:
                mat[i][j][2] = 0
                
    return mat


def get_dims(img_arr, mat):
    
    dim1_0 = []
    dim2_0 = []
    dim3_0 = []
    
    for i in range(img_arr.shape[0]):
        
        temp1_0 = 0
        temp2_0 = 0
        temp3_0 = 0
        
        for j in range(img_arr.shape[1]):
            if mat[i,j,0] == 0:
                temp1_0 += 1
                
            if mat[i,j,1] == 0:
                temp2_0 += 1

            if mat[i,j,2] == 0:
                temp3_0 += 1

        dim1_0.append(temp1_0)
        dim2_0.append(temp2_0)
        dim3_0.append(temp3_0)
        
    dim1_0 = np.array(dim1_0) / 255
    dim2_0 = np.array(dim2_0) / 255
    dim3_0 = np.array(dim3_0) / 255

    return dim1_0, dim2_0, dim3_0


def make_gcd_matrix(key):
    
    pix_val_key = []
    
    for i in range(key.shape[0]):
        for j in range(key.shape[1]):
            pix_val_key.append((key[i,j,0], key[i,j,1], key[i,j,2]))
    
    rows = key.shape[0]
    cols = key.shape[1]
    
    gcd_matrix = []

    temp_li = []
    for pix_val in pix_val_key:
        gcd1 = math.gcd(pix_val[0], pix_val[1])
        gcd2 = math.gcd(pix_val[1], pix_val[2])
        gcd3 = math.gcd(pix_val[2], pix_val[0])
        gcd = (gcd1, gcd2, gcd3)
        temp_li.append(gcd)
        if len(temp_li) % key.shape[0] == 0:
            if len(temp_li) == 0:
                temp_li = []
            else:
                gcd_matrix.append(temp_li)
                temp_li = []
          
    return gcd_matrix


def get_sum_mod(gcd_matrix):
    
    sum_mod = []
    sum_mod1 = []
    
    for i in range(len(gcd_matrix)):
        temp_sum_mod1 = 0
        temp_sum_mod2 = 0
        temp_sum_mod3 = 0
        
        for j in range(len(gcd_matrix[i])):
            if i == j:
                continue
                
            temp_sum_mod1 += gcd_matrix[i][j][0]
            temp_sum_mod2 += gcd_matrix[i][j][1]
            temp_sum_mod3 += gcd_matrix[i][j][2]
            
        temp_sum_mod = [temp_sum_mod1 % img_arr.shape[0], temp_sum_mod2 % img_arr.shape[0], temp_sum_mod3 % img_arr.shape[0]]
        
        sum_mod.append(temp_sum_mod)
    
        
    for i in range(len(gcd_matrix[0])):
        temp_sum_mod1 = 0
        temp_sum_mod2 = 0
        temp_sum_mod3 = 0
        
        for j in range(len(gcd_matrix)):
            if i == j:
                continue
                
            temp_sum_mod1 += gcd_matrix[j][i][0]
            temp_sum_mod2 += gcd_matrix[j][i][1]
            temp_sum_mod3 += gcd_matrix[j][i][2]
            
        temp_sum_mod = [temp_sum_mod1 % img_arr.shape[1], temp_sum_mod2 % img_arr.shape[1], temp_sum_mod3 % img_arr.shape[1]]
        
        sum_mod1.append(temp_sum_mod)
        
    return sum_mod, sum_mod1


def trans(sum_mod, n):
    
    trans_mod = []
    for i in range(len(sum_mod)):
        temp = []
        for j in range(len(sum_mod[i])):
            temp.append(sum_mod[i][j] % len(sum_mod))
        trans_mod.append(temp)
        
    dim1 = []
    dim2 = []
    dim3 = []
    
    for i in range(len(trans_mod)):
        if trans_mod[i][0] not in dim1:
            dim1.append(trans_mod[i][0])
        
        else:
            for j in range(trans_mod[i][0]+1, trans_mod[i][0] + n):
                if (j%n) not in dim1:
                    dim1.append(j%n)
                    break

        if trans_mod[i][1] not in dim2:
            dim2.append(trans_mod[i][1])
            
        else:
            for j in range(trans_mod[i][1]+1, trans_mod[i][1] + n):
                if (j%n) not in dim2:
                    dim2.append(j%n)
                    break

        if trans_mod[i][2] not in dim3:
            dim3.append(trans_mod[i][2])
            
        else:
            for j in range(trans_mod[i][2]+1, trans_mod[i][2] + n):
                if (j%n) not in dim3:
                    dim3.append(j%n)
                    break
                            
    return dim1, dim2, dim3

def encrypt_trans(dim1, dim2, dim3, subs_cipher_img):
    
    cipher_img = np.empty(subs_cipher_img.shape, dtype = 'uint8')
#     print(len(dim1), len(dim2), len(dim3))
    
    for i in range(subs_cipher_img.shape[0]):
        for a in range(len(dim1)):
            
            cipher_img[i, a, 0] = subs_cipher_img[i, dim1[a], 0]
            cipher_img[i, a, 1] = subs_cipher_img[i, dim2[a], 1]
            cipher_img[i, a, 2] = subs_cipher_img[i, dim3[a], 2]
            
    return cipher_img

def encrypt_trans1(dim1, dim2, dim3, subs_cipher_img):
    
    cipher_img = np.empty(subs_cipher_img.shape, dtype = 'uint8')
    
    for j in range(subs_cipher_img.shape[1]):
        for a in range(len(dim1)):

            cipher_img[a, j, 0] = subs_cipher_img[dim1[a], j, 0]
            cipher_img[a, j, 1] = subs_cipher_img[dim2[a], j, 1]
            cipher_img[a, j, 2] = subs_cipher_img[dim3[a], j, 2]

    return cipher_img

def decrypt_trans1(dim1, dim2, dim3, cipher_img):
        
    plain_img = np.empty(cipher_img.shape, dtype = 'uint8')
    
    for j in range(cipher_img.shape[1]):
        for a in range(len(dim1)):
            
            plain_img[dim1[a], j, 0] = cipher_img[a, j, 0]
            plain_img[dim2[a], j, 1] = cipher_img[a, j, 1]
            plain_img[dim3[a], j, 2] = cipher_img[a, j, 2]

    return plain_img

def decrypt_trans(dim1, dim2, dim3, cipher_img):
        
    plain_img = np.empty(cipher_img.shape, dtype = 'uint8')
    
    for i in range(cipher_img.shape[0]):
        for a in range(len(dim1)):
            
            plain_img[i, dim1[a], 0] = cipher_img[i, a, 0]
            plain_img[i, dim2[a], 1] = cipher_img[i, a, 1]
            plain_img[i, dim3[a], 2] = cipher_img[i, a, 2]

    return plain_img


def encrypt_image(img_arr, og_key):
    
    encoding_rule1 = {'00' : 'A', '11' : 'T', '01' : 'C', '10' : 'G'}
    encoding_rule2 = {'00' : 'A', '11' : 'T', '01' : 'G', '10' : 'C'}
    encoding_rule3 = {'00' : 'C', '11' : 'G', '01' : 'A', '10' : 'T'}
    encoding_rule4 = {'00' : 'G', '11' : 'C', '01' : 'A', '10' : 'T'}
    encoding_rule5 = {'10' : 'A', '01' : 'T', '00' : 'C', '11' : 'G'}
    encoding_rule6 = {'10' : 'A', '01' : 'T', '11' : 'C', '00' : 'G'}
    encoding_rule7 = {'11' : 'A', '00' : 'T', '01' : 'C', '10' : 'G'}
    encoding_rule8 = {'11' : 'A', '00' : 'T', '10' : 'C', '01' : 'G'}
    
    meta_key_bin = sum_pixels(img_arr)
    x1 = get_meta_key(og_key, meta_key_bin)

    og_key1 = left_shift(og_key, 1)

    r1 = get_meta_key(og_key1, meta_key_bin)
    r1 = (0.43 * r1) + 3.57
    
    key = keygen(x1, r1, img_arr.shape[0] * img_arr.shape[1] * img_arr.shape[2])

    gcd_matrix = make_gcd_matrix(key)
    sum_mod, sum_mod1 = get_sum_mod(gcd_matrix)
    dim1h, dim2h, dim3h = trans(sum_mod, img_arr.shape[1])
    dim1v, dim2v, dim3v = trans(sum_mod1, img_arr.shape[0])
    cipher_img_trans = img_arr.copy()

    cipher_img_trans = encrypt_trans(dim1h, dim2h, dim3h, cipher_img_trans)
    cipher_img_trans = encrypt_trans1(dim1v, dim2v, dim3v, cipher_img_trans)
    
    og_key_l = left_shift(og_key1[:32], 1)
    og_key_lm = left_shift(og_key1[32:64], 2)
    og_key_rm = left_shift(og_key1[64:96], 3)
    og_key_r = left_shift(og_key1[96:128], 4)
    og_key2 = og_key_l + og_key_lm + og_key_rm + og_key_r
    r2 = int(og_key2, 2) / (math.pow(2, 128))
    r2 = (0.43 * r2) + 3.57
    
    og_key_l = right_shift(og_key[:64], 2)
    og_key_r = right_shift(og_key[64:], 1)
    og_key3 = og_key_l + og_key_r
    x2 = int(og_key3, 2) / (math.pow(2, 128))
    
    key = keygen(x2, r2, img_arr.shape[0] * img_arr.shape[1] * img_arr.shape[2])
    
    first_dim, second_dim, third_dim = decToBinary_dims(key)
    mat = get_mat(img_arr, encoding_rule1, first_dim, second_dim, third_dim)
    dim1_0, dim2_0, dim3_0 = get_dims(img_arr, mat)
    
    cipher_img = np.empty(img_arr.shape, dtype = 'uint8')
    
    for i in range(img_arr.shape[0]):

        xkey, ykey, zkey = Lorenzkey(dim1_0[i], dim2_0[i], dim3_0[i], img_arr.shape[1] + 10)
        xkey, ykey, zkey = xkey[6:-5], ykey[6:-5], zkey[6:-5]
        
        j = 0
        for x in xkey:
            cipher_img[i,j,0] = cipher_img_trans[i,j,0] ^ int(x)
            j += 1

        j = 0
        for y in ykey:
            cipher_img[i,j,1] = cipher_img_trans[i,j,1] ^ int(y)
            j += 1

        j = 0
        for z in zkey:
            cipher_img[i,j,2] = cipher_img_trans[i,j,2] ^ int(z)
            j += 1
            
    meta_key_bin1 = sum_pixels(cipher_img) 
    x3 = get_meta_key(og_key, meta_key_bin1)
    r3 = get_meta_key(og_key1, meta_key_bin1)
    r3 = (0.43 * r3) + 3.57
    
    key = keygen(x3, r3, img_arr.shape[0] * img_arr.shape[1] * img_arr.shape[2])
    gcd_matrix = make_gcd_matrix(key)
    sum_mod, sum_mod1 = get_sum_mod(gcd_matrix)

    dim1h, dim2h, dim3h = trans(sum_mod, img_arr.shape[1])
    dim1v, dim2v, dim3v = trans(sum_mod1, img_arr.shape[0])
    cipher_img_trans = cipher_img.copy()
   
    cipher_img_trans = encrypt_trans(dim1h, dim2h, dim3h, cipher_img)

    cipher_img_trans = encrypt_trans1(dim1v, dim2v, dim3v, cipher_img_trans)

    return cipher_img_trans


def decrypt_image(cipher_img, og_key):
    
    encoding_rule1 = {'00' : 'A', '11' : 'T', '01' : 'C', '10' : 'G'}
    encoding_rule2 = {'00' : 'A', '11' : 'T', '01' : 'G', '10' : 'C'}
    encoding_rule3 = {'00' : 'C', '11' : 'G', '01' : 'A', '10' : 'T'}
    encoding_rule4 = {'00' : 'G', '11' : 'C', '01' : 'A', '10' : 'T'}
    encoding_rule5 = {'10' : 'A', '01' : 'T', '00' : 'C', '11' : 'G'}
    encoding_rule6 = {'10' : 'A', '01' : 'T', '11' : 'C', '00' : 'G'}
    encoding_rule7 = {'11' : 'A', '00' : 'T', '01' : 'C', '10' : 'G'}
    encoding_rule8 = {'11' : 'A', '00' : 'T', '10' : 'C', '01' : 'G'}
    
    meta_key_bin1 = sum_pixels(cipher_img)

    og_key1 = left_shift(og_key, 1)
    
    og_key_l = left_shift(og_key1[:32], 1)
    og_key_lm = left_shift(og_key1[32:64], 2)
    og_key_rm = left_shift(og_key1[64:96], 3)
    og_key_r = left_shift(og_key1[96:128], 4)
    og_key2 = og_key_l + og_key_lm + og_key_rm + og_key_r
    
    og_key_l = right_shift(og_key[:64], 2)
    og_key_r = right_shift(og_key[64:], 1)
    og_key3 = og_key_l + og_key_r
    
    x3 = get_meta_key(og_key, meta_key_bin1)
    r3 = get_meta_key(og_key1, meta_key_bin1)
    r3 = (0.43 * r3) + 3.57
    
    key = keygen(x3, r3, img_arr.shape[0] * img_arr.shape[1] * img_arr.shape[2])
    
    gcd_matrix = make_gcd_matrix(key)
    sum_mod, sum_mod1 = get_sum_mod(gcd_matrix)

    dim1h, dim2h, dim3h = trans(sum_mod, img_arr.shape[1])
    dim1v, dim2v, dim3v = trans(sum_mod1, img_arr.shape[0])
    cipher_img_trans = cipher_img.copy()
    
    cipher_img_trans = decrypt_trans1(dim1v, dim2v, dim3v, cipher_img)

    cipher_img_trans = decrypt_trans(dim1h, dim2h, dim3h, cipher_img_trans)
    
    cipher_img = cipher_img_trans.copy()
    
    r2 = int(og_key2, 2) / (math.pow(2, 128))
    r2 = (0.43 * r2) + 3.57
    x2 = int(og_key3, 2) / (math.pow(2, 128))
    
    key = keygen(x2, r2, cipher_img.shape[0] * cipher_img.shape[1] * cipher_img.shape[2])
    
    first_dim, second_dim, third_dim = decToBinary_dims(key)
    mat = get_mat(cipher_img, encoding_rule1, first_dim, second_dim, third_dim)
    dim1_0, dim2_0, dim3_0 = get_dims(cipher_img, mat)
    
    plain_img = np.empty(cipher_img.shape, dtype = 'uint8')

    for i in range(cipher_img.shape[0]):
        
        xkey, ykey, zkey = Lorenzkey(dim1_0[i], dim2_0[i], dim3_0[i], cipher_img.shape[1] + 10)
        xkey, ykey, zkey = xkey[6:-5], ykey[6:-5], zkey[6:-5]
        
        j = 0
        for x in xkey:
            plain_img[i,j,0] = cipher_img[i,j,0] ^ int(x)
            j += 1

        j = 0
        for y in ykey:
            plain_img[i,j,1] = cipher_img[i,j,1] ^ int(y)
            j += 1

        j = 0
        for z in zkey:
            plain_img[i,j,2] = cipher_img[i,j,2] ^ int(z)
            j += 1
            
    meta_key_bin = sum_pixels(plain_img)
    x1 = get_meta_key(og_key, meta_key_bin)
    r1 = get_meta_key(og_key1, meta_key_bin)
    r1 = (0.43 * r1) + 3.57

    key = keygen(x1, r1, cipher_img.shape[0] * cipher_img.shape[1] * cipher_img.shape[2])
    
    gcd_matrix = make_gcd_matrix(key)
    sum_mod, sum_mod1 = get_sum_mod(gcd_matrix)
    dim1h, dim2h, dim3h = trans(sum_mod, img_arr.shape[1])
    dim1v, dim2v, dim3v = trans(sum_mod1, img_arr.shape[0])
    
    plain_img = decrypt_trans1(dim1v, dim2v, dim3v, plain_img)
    plain_img = decrypt_trans(dim1h, dim2h, dim3h, plain_img)
    
    return plain_img


og_key = input("Provide 128-bits key in binary format: ")
image_path = input("Provide the path of the image: ")
output_path = input("Provide the path to save the output image: ")
process = int(input("Press 1 to encrypt the image and 2 to decrypt the image: "))

img = PIL.Image.open(image_path, 'r')
img_arr = np.array(img)

if process == 1:
	cipher_img = encrypt_image(img_arr, og_key)
	ci_im = Image.fromarray(cipher_img)
	ci_im.save(output_path + '/encrypted.png')

elif process == 2:
	plain_img = decrypt_image(img_arr, og_key)
	pl_im = Image.fromarray(plain_img)
	pl_im.save(output_path + '/decrypted.png')

else:
	print("Invalid input!")


