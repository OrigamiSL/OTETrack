import os
import argparse



def process_nfs(root_path):
    # root_path = '/home/lhg/work/data/track_data/data/Nfs'
    file_list = os.listdir(root_path)

    for file in file_list:
        anno_path = os.path.join(root_path,file,file,'30',file+'.txt')
        new_path = os.path.join(root_path,file,file,'30',file+'_n1_biaozhun'+'.txt')
        # new_path = os.path.join(root_path,file,file,'30',file+'_test2'+'.txt')
        with open(anno_path,'r') as f:
            h = f.readlines()
            for i in range(len(h)):
                print(i)
                if i == 0 or (i+1) % 8 == 0:
                    x = h[i].split('\t')
                    x1 = x[0]
                    # print(x1)
                    xx = x1.split(' ')[1]
                    yy = x1.split(' ')[2]
                    x2 = str(int(x1.split(' ')[3])-int(x1.split(' ')[1]))
                    y2 = str(int(x1.split(' ')[4])-int(x1.split(' ')[2]))
                    new_line = xx+'\t'+yy+'\t'+x2+'\t'+y2
                    # print(new_line)
                    with open(new_path,'a') as ff:
                        ff.write(new_line)
                        ff.write('\n')

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--root_path', type=str, help='')

    args = parser.parse_args()

    process_nfs(args.root_path)


if __name__ == '__main__':
    main()