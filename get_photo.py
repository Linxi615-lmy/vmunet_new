import os
import shutil

def extract_images_from_txt(image_folder, txt_file, output_folder="isic2017_test"):
    """
    根据txt文件中的图片名称，从源文件夹提取对应图片并保存到目标文件夹。
    
    :param image_folder: 包含所有原始图片的文件夹路径
    :param txt_file: 包含要提取的图片名称的txt文件路径（每行一个名称）
    :param output_folder: 提取出来的图片保存位置，默认为 'isic2017_test'
    """
    # 如果输出路径不存在，则自动创建该目录
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 读取txt文件中的图片名内容
    with open(txt_file, 'r', encoding='utf-8') as f:
        # 去除换行符和空行
        image_names = [line.strip() for line in f.readlines() if line.strip()]

    success_count = 0
    missing_count = 0

    for name in image_names:
        src_path = os.path.join(image_folder, name)
        
        # 1. 如果txt里带了后缀，并且文件刚好存在
        if os.path.exists(src_path):
            dst_path = os.path.join(output_folder, name)
            shutil.copy(src_path, dst_path)
            success_count += 1
        else:
            # 2. 如果txt里没有带有后缀，尝试用常见的后缀匹配
            found = False
            for ext in ['.jpg', '.png', '.jpeg']:
                img_with_ext = name + ext
                src_with_ext = os.path.join(image_folder, img_with_ext)
                if os.path.exists(src_with_ext):
                    dst_with_ext = os.path.join(output_folder, img_with_ext)
                    shutil.copy(src_with_ext, dst_with_ext)
                    success_count += 1
                    found = True
                    break
            
            if not found:
                print(f"未找到图片: {name}")
                missing_count += 1

    print(f"提取任务完成！成功复制：{success_count} 张，未找到：{missing_count} 张。")


# ==========================================
# 使用示例：
# ==========================================
if __name__ == '__main__':
    # 替换成你实际的文件夹路径和txt路径
    SOURCE_IMAGE_DIR = "/media/data/liumengyu/Dataset/MedicalImageDataset/ISIC17/allimg/" 
    TXT_PATH = "data_new/ISIC2017/test.txt"
    OUTPUT_DIR = "isic2017_test"
    
    extract_images_from_txt(SOURCE_IMAGE_DIR, TXT_PATH, OUTPUT_DIR)