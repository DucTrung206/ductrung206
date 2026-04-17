
## Giới thiệu dự án 
Tên dự án: Nhận diện hình ảnh qua chuyển động và trạng thái tay

Dự án này ứng dụng công nghệ nhận diện hình ảnh của OpenCV và MediaPipe Hands để theo dõi chuyển động và trạng thái của các ngón tay. Thay vì sử dụng những khung hình chữ nhật nhàm chán, hệ thống có khả năng tính toán toán học để tự động tạo ra các hình khối đa giác linh hoạt bao quanh các ngón tay của bạn.
Sau khi người dùng chốt khung hình bằng cách giữ yên tay, ứng dụng sẽ thực hiện kỹ thuật cắt lớp ảnh (Masking & Alpha Blending) để đóng băng vạn vật xung quanh, chỉ để lại một "cánh cửa" video chuyển động trực tiếp bên trong lòng bàn tay bạn.
## Thuật toán, thư viện sử dụng trong dự án
Ngôn ngữ chính : Pyhon
Thư viện sử dụng: OpenCV(xử lý video, nhận diện vật thể, chuyển động của vật thể), MediaPipe Hands(sử dụng AI để nhận diện và theo dõi bàn tay),time(thời gian chờ),math(tính toán)
##  Tính năng nổi bật

-  **Nhận diện trạng thái Mở/Khép ngón tay:** Thuật toán thông minh tính toán khoảng cách từ đầu ngón tay và khớp ngón tay đến cổ tay để phân loại chính xác ngón nào đang Mở (Xanh lá) và Khép (Đỏ).
-  **Tạo hình linh hoạt (Convex Hull):** - **1 Ngón:** Không tạo hình, cảnh báo chấm Đỏ, 2 ngón không hiện vẽ
  - **3-5 Ngón:** Sử dụng thuật toán Convex Hull để bọc màng lưới (Polygon) bao quanh các ngón tay.
- **Chụp ảnh bằng cử chỉ (Anti-Shake):** Thuật toán tính toán độ lệch pixel. Người dùng chỉ cần giữ yên khung tay trong `2.5 giây`, thanh Loading sẽ chạy và tự động "Chụp", khi đưa tay lên lập tức kết thúc tiến trình đóng băng và nhận diện các ngón tay mở có thể tạo thành hình và tiếp tục lập lại các thao tác trên.
- **Hiệu ứng Magic Masking:** Khi chụp xong, nền video sẽ bị đóng băng, xuất hiện vòng tròn ma thuật màu tím bao quanh, và chỉ phần lõi đa giác là giữ lại hình ảnh Live Camera.
## Yêu cầu hệ thống: 
Yêu cầu hệ thống có python 3.8 trở lên và đã cài sẵn 2 thư viện OpenCV và MediaPipe Hands
## Cách sử dụng:

B1: Đưa bàn tay lên trước Camera.
B2: Xoè / Cụp các ngón tay để thấy hệ thống nhận diện điểm xanh/đỏ và tự động thay đổi hình khối.
B3: Khi đã ưng ý với một hình khối, giữ yên bàn tay.
B4: Thanh tiến trình màu xanh lá sẽ chạy lên cùng dòng chữ "Dang giu yen...". Đợi 2.5 giây. Khung hình chuyển sang màu vàng có vòng tròn tím. Giờ bạn có thể bỏ tay xuống. Hiệu ứng sẽ duy trì trong 20 giây. Nếu muốn vẽ hình khác, chỉ cần giơ tay lên trước màn hình một lần nữa, hệ thống sẽ tự động Reset!
B5: Nhấn nút ESC để thoát chương trình.
## Cấu trúc thuật toán chính
get_polygon_from_points(): Xử lý dữ liệu điểm ảnh đầu vào và xuất ra mảng đa giác lồi bọc ngoài cùng.
cv2.fillPoly() & np.where(): Kỹ thuật tách nền nâng cao không cần vòng lặp for, tối ưu hóa tốc độ CPU.
cv2.LINE_AA: Khử răng cưa cho các đối tượng hình học.



