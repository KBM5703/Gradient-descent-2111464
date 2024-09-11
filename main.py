import os

if __name__ == '__main__':
    while True:
        print("\nChọn một tùy chọn:")
        print("1. Đăng ký khuôn mặt mới")
        print("2. Nhận diện khuôn mặt")
        print("3. Thoát")

        choice = input("Nhập lựa chọn của bạn (1/2/3): ")

        if choice == '1':
            os.system('python registration.py')
        elif choice == '2':
            os.system('python recognition.py')
        elif choice == '3':
            break
        else:
            print("Lựa chọn không hợp lệ. Vui lòng thử lại.")
