#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

Adafruit_MPU6050 mpu;

void setup() {
  Serial.begin(115200);
  Serial.println("Initializing MPU6050...");

  // Initialize I2C communication
  Wire.begin(21, 22); // SDA = GPIO 21, SCL = GPIO 22

  // Initialize MPU6050
  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip. Check connections.");
    while (1) {
      delay(10);
    }
  }
  Serial.println("MPU6050 initialized successfully!");

  // Configure the sensor (optional)
  mpu.setAccelerometerRange(MPU6050_RANGE_2_G);
  mpu.setGyroRange(MPU6050_RANGE_250_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_5_HZ);
}

void loop() {
  // Get new sensor event
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  // Print accelerometer values
  Serial.print("Accel X: "); Serial.print(a.acceleration.x); Serial.print(" m/s^2, ");
  Serial.print("Y: "); Serial.print(a.acceleration.y); Serial.print(" m/s^2, ");
  Serial.print("Z: "); Serial.print(a.acceleration.z); Serial.println(" m/s^2");

  // Print gyroscope values
  Serial.print("Gyro X: "); Serial.print(g.gyro.x); Serial.print(" rad/s, ");
  Serial.print("Y: "); Serial.print(g.gyro.y); Serial.print(" rad/s, ");
  Serial.print("Z: "); Serial.print(g.gyro.z); Serial.println(" rad/s");

  // Print temperature
  Serial.print("Temperature: "); Serial.print(temp.temperature); Serial.println(" °C");

  // Delay for readability
  delay(500);
}
