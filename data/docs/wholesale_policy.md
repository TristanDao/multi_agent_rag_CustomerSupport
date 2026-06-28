# Chính Sách Bán Sỉ (Wholesale Policy)

Tài liệu này mô tả các quy tắc áp dụng cho khách hàng mua sỉ, khách hàng doanh nghiệp và nhà phân phối của cửa hàng văn phòng phẩm.

## 1. Điều Kiện Mua Sỉ

- Khách hàng được xếp vào nhóm **wholesale** khi đơn hàng đạt tổng giá trị tối thiểu **5.000.000 VNĐ** trong vòng 30 ngày gần nhất, hoặc khi đăng ký theo hợp đồng đối tác.
- Khách hàng **distributor** phải ký hợp đồng phân phối chính thức trước khi áp dụng giá distributor.
- Khách hàng **corporate** được áp dụng giá sỉ ngay từ đơn đầu tiên theo thỏa thuận hợp đồng khung.

## 2. Bậc Giá Theo Số Lượng

Áp dụng các mức chiết khấu theo số lượng cho mỗi SKU:

- **1 - 9 đơn vị**: giá bán lẻ cơ bản.
- **10 - 49 đơn vị**: giảm thêm **3 - 5%** so với giá bán lẻ.
- **50 - 199 đơn vị**: giảm thêm **7 - 12%**.
- **≥ 200 đơn vị**: giảm thêm **15 - 20%**, có thể thương lượng trực tiếp với quản lý.

Giá chính thức được lưu trong bảng `price_tiers` theo `customer_type` và `min_quantity`.

## 3. Chính Sách Khuyến Mãi

- Các chương trình khuyến mãi **không cộng dồn** (non-stackable): mỗi đơn hàng chỉ áp dụng **một** mã giảm giá hoặc chương trình khuyến mãi duy nhất.
- Khuyến mãi theo mã (promo code) có thời hạn cụ thể, xem chi tiết tại `promotions`.
- Khuyến mãi không áp dụng cho các sản phẩm đặc biệt đã có giá distributor.

## 4. Phê Duyệt Đặc Biệt

- Các thỏa thuận giá đặc biệt ngoài bảng giá cần được **quản lý bán hàng phê duyệt** trước khi xuất đơn.
- Đơn hàng từ **20.000.000 VNĐ** trở lên cần duyệt bởi **giám đốc chi nhánh**.
- Các chương trình hợp tác dài hạn với khách corporate/distributor cần ký phụ lục hợp đồng.

## 5. Thanh Toán Và Công Nợ

- Khách hàng mới cần đặt cọc tối thiểu **30%** giá trị đơn wholesale đầu tiên.
- Khách hàng wholesale có thể được cấp hạn mức tín dụng sau 3 đơn thanh toán đúng hạn.
- Điều khoản thanh toán tiêu chuẩn: **NET 7, NET 15, hoặc NET 30** tùy hợp đồng.

## 6. Đổi Trả Hàng Sỉ

- Hàng sỉ chỉ được đổi trả khi có lỗi từ nhà sản xuất hoặc giao sai mã SKU.
- Hàng sỉ không áp dụng chính sách đổi trả 7 ngày như bán lẻ, trừ khi có thỏa thuận khác bằng văn bản.
