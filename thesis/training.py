import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from tensorflow.keras.models import Model

# 1. 데이터 경로 설정
IMAGE_DIR = 'path_to_images'  # 도로 이미지 디렉토리
MASK_DIR = 'path_to_masks'    # 차선 마스크 디렉토리

# 2. 파일 리스트 가져오기
image_files = sorted([os.path.join(IMAGE_DIR, file) for file in os.listdir(IMAGE_DIR) if file.endswith('.jpg') or file.endswith('.png')])
mask_files = sorted([os.path.join(MASK_DIR, file) for file in os.listdir(MASK_DIR) if file.endswith('.jpg') or file.endswith('.png')])

# 3. 데이터셋 분할
train_images, val_images, train_masks, val_masks = train_test_split(image_files, mask_files, test_size=0.2, random_state=42)

# 4. 데이터 제너레이터 정의
class DataGenerator(Sequence):
    def __init__(self, images, masks, batch_size=8, img_size=(256, 256), augment=False):
        self.images = images
        self.masks = masks
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.images[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.masks[idx * self.batch_size:(idx + 1) * self.batch_size]

        images = []
        masks = []

        for img_path, mask_path in zip(batch_x, batch_y):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
            img = img / 255.0

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, self.img_size)
            _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

            images.append(img)
            masks.append(mask)

        images = np.array(images, dtype=np.float32)
        masks = np.array(masks, dtype=np.float32)
        masks = np.expand_dims(masks, axis=-1)

        if self.augment:
            for i in range(len(images)):
                if np.random.rand() > 0.5:
                    images[i] = np.fliplr(images[i])
                    masks[i] = np.fliplr(masks[i])

        return images, masks

# 5. 제너레이터 인스턴스 생성
train_gen = DataGenerator(train_images, train_masks, batch_size=8, img_size=(256, 256), augment=True)
val_gen = DataGenerator(val_images, val_masks, batch_size=8, img_size=(256, 256), augment=False)

# 6. U-Net 모델 정의
def unet_model(input_size=(256, 256, 3)):
    inputs = Input(input_size)

    # 인코더
    c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)

    c2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D(pool_size=(2, 2))(c2)

    c3 = Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = MaxPooling2D(pool_size=(2, 2))(c3)

    c4 = Conv2D(512, 3, activation='relu', padding='same')(p3)
    c4 = Conv2D(512, 3, activation='relu', padding='same')(c4)
    drop4 = Dropout(0.5)(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    # Bottleneck
    c5 = Conv2D(1024, 3, activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, 3, activation='relu', padding='same')(c5)
    drop5 = Dropout(0.5)(c5)

    # 디코더
    u6 = UpSampling2D(size=(2, 2))(drop5)
    u6 = Conv2D(512, 2, activation='relu', padding='same')(u6)
    merge6 = concatenate([drop4, u6], axis=3)
    c6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
    c6 = Conv2D(512, 3, activation='relu', padding='same')(c6)

    u7 = UpSampling2D(size=(2, 2))(c6)
    u7 = Conv2D(256, 2, activation='relu', padding='same')(u7)
    merge7 = concatenate([c3, u7], axis=3)
    c7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
    c7 = Conv2D(256, 3, activation='relu', padding='same')(c7)

    u8 = UpSampling2D(size=(2, 2))(c7)
    u8 = Conv2D(128, 2, activation='relu', padding='same')(u8)
    merge8 = concatenate([c2, u8], axis=3)
    c8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
    c8 = Conv2D(128, 3, activation='relu', padding='same')(c8)

    u9 = UpSampling2D(size=(2, 2))(c8)
    u9 = Conv2D(64, 2, activation='relu', padding='same')(u9)
    merge9 = concatenate([c1, u9], axis=3)
    c9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
    c9 = Conv2D(64, 3, activation='relu', padding='same')(c9)

    outputs = Conv2D(1, 1, activation='sigmoid')(c9)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# 7. 모델 컴파일
model = unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 8. 모델 학습
checkpoint = tf.keras.callbacks.ModelCheckpoint('unet_lane_detection.h5', verbose=1, save_best_only=True)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    callbacks=[checkpoint, early_stop]
)

# 9. 학습 결과 시각화
plt.figure(figsize=(12, 4))

# 손실 그래프
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Binary Crossentropy')
plt.legend()

# 정확도 그래프
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# 10. 모델 평가 및 예측
def predict_lane(image_path, model, img_size=(256, 256)):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, img_size)
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)

    pred_mask = model.predict(img_input)[0]
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    pred_mask_resized = cv2.resize(pred_mask, (img.shape[1], img.shape[0]))

    lane_image = img_rgb.copy()
    lane_image[pred_mask_resized == 255] = [255, 0, 0]  # 차선을 빨간색으로 표시

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(lane_image)
    plt.title("Lane Detection")
    plt.axis('off')

    plt.show()

# 예측 실행
sample_image_path = 'path_to_sample_image.jpg'  # 예측할 이미지 경로
predict_lane(sample_image_path, model)
