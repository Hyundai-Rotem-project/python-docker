from StereoImageFilter import StereoImageFilter
# 이미지 파일이 있는 디렉토리 경로
left_folder = "C:/Users/Dhan/Documents/Tank Challenge/capture_images/L"
right_folder = "C:\\Users\\Dhan\\Documents\\Tank Challenge\\capture_images\\R"
log_path = "C:\\Users\\Dhan\\Documents\\Tank Challenge\\log_data\\tank_info_log.txt"

filter = StereoImageFilter(left_folder,right_folder,log_path)

log_filtered = filter.get_result()
print(log_filtered)