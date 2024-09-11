Rain_SENSOR_PIN = 27

GPIO.setmode(GPIO.BCM)

GPIO.setup(SERVO_PIN, GPIO.OUT)
GPIO.setup(Rain_SENSOR_PIN, GPIO.IN)

pwm = GPIO.PWM(SERVO_PIN, 50)
pwm.start(7.5)

try:
    while True:
        # Read IR sensor value
        if GPIO.input(Rain_SENSOR_PIN) == 1:  # Object detected
            print("Rain detected!")
            pwm.ChangeDutyCycle(12.5)  # Move servo to 90 degrees
            time.sleep(1)
            pwm.ChangeDutyCycle(7.5)  # Move servo back to 0 degrees

except KeyboardInterrupt:
    pwm.stop()
    GPIO.cleanup()
    print("Program stopped!")
