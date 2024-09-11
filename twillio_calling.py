import os
from twilio.rest import Client
import RPi.GPIO as GPIO
import time
import dotenv

dotenv.load_dotenv()

# Set up GPIO using BCM numbering
GPIO.setmode(GPIO.BCM)

# Set up the GPIO pin 17 as an input with a pull-down resistor
GPIO.setup(17, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(18, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

# Twilio account SID and Auth Token
account_sid = os.getenv("TWILLIO_ACCOUNT_SID")
auth_token = os.getenv("TWILLIO_AUTH_TOKEN")
client = Client(account_sid, auth_token)

try:
    while True:
        input_state = GPIO.input(17)  # Read the state of the input pin
        input_state_18 = GPIO.input(18)
        
        if input_state == GPIO.HIGH:  # Trigger when 3.3V is applied
            # Twilio call logic when the button is pressed
            call = client.calls.create(
               url="https://plugsi.live/gas.xml",
               to=os.getenv("TWILLIO_TO"),
               from_=os.getenv("TWILLIO_FROM"),
            )
            print("call gas")
            print(f"Call initiated from pin 17 with SID: {call.sid}")
            time.sleep(5*60)  # Add a delay to prevent multiple calls due to button bounce
        elif input_state_18 == GPIO.HIGH:  # Trigger when 3.3V is applied to pin 18
             # Twilio call logic when the button connected to pin 18 is pressed
            call = client.calls.create(
                url="https://plugsi.live/fire.xml",  # Change the URL for pin 18
                to=os.getenv("TWILLIO_TO"),
                from_=os.getenv("TWILLIO_FROM"),
            )
            print("call fire")
            print(f"Call initiated from pin 18 with SID:, {call.sid}")
            time.sleep(5*60)  # Add a delay to prevent multiple calls   
        time.sleep(0.1)  # Delay to debounce the button

except KeyboardInterrupt:
    pass

finally:
    GPIO.cleanup()  # Clean up GPIO on exit
