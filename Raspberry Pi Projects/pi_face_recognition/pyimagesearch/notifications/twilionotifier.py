# import the necessary packages
from twilio.rest import TwilioRestClient
from boto.s3.connection import S3Connection
from boto.s3.key import Key
from threading import Thread
from imutils.io import TempFile
import cv2

class TwilioNotifier:
	def __init__(self, conf):
		# store the configuration object
		self.conf = conf

	def send(self, image):
		# create a temporary path for the image and write it to file
		tempImage = TempFile()
		cv2.imwrite(tempImage.path, image)

		# start a thread to upload the file and send it
		t = Thread(target=self._send, args=(tempImage,))
		t.daemon = True
		t.start()

	def _send(self, tempImage):
		# connect to S3 and grab the bucket
		s3 = S3Connection(self.conf["aws_access_key_id"], self.conf["aws_secret_access_key"])
		bucket = s3.get_bucket(self.conf["s3_bucket"])

		# upload the file, make it public, and generate a URL for the file
		k = Key(bucket)
		k.key = tempImage.path[tempImage.path.rfind("/") + 1:]
		k.set_contents_from_filename(tempImage.path)
		k.make_public()
		url = k.generate_url(expires_in=300)

		# connect to Twilio and send the file via MMS
		client = TwilioRestClient(self.conf["twilio_sid"], self.conf["twilio_auth"])
		client.messages.create(to=self.conf["twilio_to"], from_=self.conf["twilio_from"],
			body=self.conf["message_body"], media_url=url)

		# delete the temporary file
		tempImage.cleanup()