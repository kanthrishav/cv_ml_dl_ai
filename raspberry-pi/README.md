# HARDWARE, SETUP AND INSTALLATIONS

This is the setup guide for Raspberry Pi 5 8 GB running the recommended version of Bookworm OS, which is meant for only python based development of computer vision tasks. For C++ based usage, I may create a different setup guide due to the compatibility issues between multiple libraries.

List of Hardware components used for this task
1. Raspberry Pi 5 Model B 8 GB RAM
2. Raspberry Pi 5 Official 27W Power Supply PD Adapter 5.1V 5A USB-C
3. Active Cooler for Raspberry 5 with 30mm PWM 4-Pin Cooling Fan Pi 5 Heatsink with Thermal Tape
4. microSD card (64 GB) Class 10, UHS-I, U1 Card
5. Camera - Raspberry Pi 5Mp AGP Gddr3 Camera Board Module (basic one)
6. Camera - Official Raspberry Pi AI Camera with SONY IMX500 Sensor
7. Camera - Raspberry Pi High Quality Camera with Interchangeable Lens Base
8. Raspberry Pi Camera Cable Standard - Mini - 300mm
9. USB microSD card reader
10. Laptop (Windows)

(Since I am located in India, most of the products were purchased from either Robocraze or Amazon)

Overall sequence of steps used for the setup are as follows : 
	1. Flashing the OS (on laptop)
	2. First boot up (on RPI)
	3. Setting up VNC (on RPI)
	4. Libraries installations
	5. Testing the camera

# FLASHING THE OS (on the Laptop)
1. Insert the microSD in the USB card reader which is plugged into your laptop, and format it.
2. Download and install the free Raspberry Pi Imager Software from the official website of RPI https://www.raspberrypi.com/software/
3. Launch the RPI Imager
	<img width="855" height="602" alt="image" src="https://github.com/user-attachments/assets/520314f9-ea3a-478e-a395-42d17599bf38" />

 1. Click "Choose Device" -> select "Raspberry Pi 5, 500 and Compute Module 5"
 2. Click "Choose OS" ->  select "Respberry Pi IS (64 -bit) :  A port of Debian Bookworm with the Raspberry Pi Desktop (Recommended)"
 3. Click "Choose Storage" -> select your microSD card (sometimes, it takes time for the Imager to find your microSD card mount)
 4. Click Next
 5. Upon clicking next you will see these options
    
 	<img width="657" height="203" alt="image" src="https://github.com/user-attachments/assets/1ac3c9bb-07af-4d6b-bb47-73d559f63313" />

  
 6. Click on "Edit Settings" - > Three tabs will be there General, Services and Options
	1) In the General Tab check the options

		<img width="1125" height="1035" alt="image" src="https://github.com/user-attachments/assets/6501c704-1b27-4f65-ac18-4a30c2ad19fb" />
  
		a) Check Set hostname : then provide a name for your pi (you can choose anything). Lets say we used "pi" as the hostname
		b) Check Set username and password : username may or may not same as the hostname (up to you), provide a password. Lets say we used "pi" as the username as well.
		c) Check Configure wireless LAN : SSID is the name of your Wifi, so provide SSID and wifi password
		d) Check locale settings : these usually get populated on its own, if not you can make your selection
	2) In the Services tab

		<img width="772" height="347" alt="image" src="https://github.com/user-attachments/assets/cb264ab8-cce4-4ffa-93f0-9a0bd615c560" />

		a) Check "Enable SSH"
		b) You can either go with password authentication or allow public-key authentication only
			i) For public key authentication only, you will need to generate a public key
	3) In the Options tab
	
  		<img width="1126" height="423" alt="image" src="https://github.com/user-attachments/assets/7d997883-e9f9-488d-8b42-dc544ab937ea" />

		a) These will be checked by default, if you can check them as well.
	4) Click "SAVE"
	5) Then you will get a warning saying that all the data on your microSD will be erased, go ahead and press Continue
	6) The Imager will start writing the OS first, then it will start verifying
		a) This process is going to take some time. For me it was usually 45 mins
		b) Do not click cancel during the process or allow your laptop to get turned off : this may cause corruption of your microSD card
		c) It is going to use the internet to fetch the OS image, so don’t let your internet connection to drop as well. If there is chance that your internet connection may drop
			i) Download the OS image separately from the website
			ii) Raspberry Pi OS downloads – Raspberry Pi
			iii) And then in the imager provide the path to this locally downloaded image file
			iv) If it does get interrupted due to internet, you can restart the process and try again. I wouldn’t suggest formatting it yourself again. Better to let the Imager format the card. But if that doesn’t work, format it manually and then restart.
	7) Wait for it to finish
	
	
# FIRST BOOTUP (on RPI)
	
1. Insert the microSD card with the flashed OS into RPI
2. Connect the power cable
3. We have already setup wifi so you do not need Ethernet LAN connection, but if you want, you can attach that too
4. We have already enabled SSH so you do not need a separate display. But if you want you can attach a display using a microHDMI to HDMI cable, plus keyboard and mouse. 
5. Download and install an IP scanner software like Angry IP Scanner.
	
 	<img width="1176" height="921" alt="image" src="https://github.com/user-attachments/assets/7d02c7b9-fd4a-4669-9392-fb1553373176" />

    i. Click "Start"
	ii. It will start searching for all the devices currently present in your network within the default subnet range given under "IP Range"
    iii. After the search ends, find your RPI with the hostname that you had given and make a note of the IP address of your RPI
   
   	<img width="1221" height="1081" alt="image" src="https://github.com/user-attachments/assets/f89e50d0-7490-4c64-b9ae-89755b91a347" />
			
	iv. Alternatively, instead of using a third party software like the Angry IIP Scanner, you can search for the IP of your RPI using either your router's DHCP client list or by running this command on your terminal in your laptop "namp -sP 192.168.0.0/255"
    v. Lets say you found that the IP address of your RPI is "192.168.0.105"
6. Open a terminal in your laptop (preferably MobaXterm in Windows)
	i. Run the command "ssh pi2pi.local" or "ssh pi@192.168.0.102"
	ii. Enter the password
	iii. And you will be in

    <img width="952" height="303" alt="image" src="https://github.com/user-attachments/assets/2b8c52c4-5415-4a53-9ae5-a35ca96fbb3f" />

7. Run "sudo apt update && sudo apt full-upgrade -y"
	i. And once this ends, reboot your RPI using "sudo reboot"

	
# SETTING UP VNC
	
1. Once the RPI is up, in your laptop terminal run "sudo raspi-config"
2. Select "Interface Options" -> Enable VNC (or something like that) -> Enable -> Yes
3. Download & install RealVNC in your laptop
	i. Once installed, open the RealVNC Viewer
	ii. On the top left corner, click File -> New connection
			
	<img width="1117" height="809" alt="image" src="https://github.com/user-attachments/assets/be8f00a3-dcfd-40ff-85e3-a5ed50f99636" />

 	<img width="583" height="796" alt="image" src="https://github.com/user-attachments/assets/1e09b1e2-72e6-4978-8e4f-806f4a8677a3" />

	iii. Enter the hostname or the IP address of your RPI
	iv. You can give it a Name which is just a name to identify the device within RealVNC alone
	v. Make sure that the options that you see checked off in the images are checked off for you too
	vi. And then press ok
	vii. You will see your pi added as a connection in the RealVNC Viewer like this
	
 	<img width="249" height="208" alt="image" src="https://github.com/user-attachments/assets/2b26ae7a-5b82-4ee2-9db9-c717ac1e500f" />

    viii. Right click on it -> Connect
	ix. And you will able to view the desktop of your RPI in the RealVNC Viewer
	
	
# LIBRARIES INSTALLATION (Bare minimum but really clean)
	
In this section I will stick to installing only those libraries that are absolutely needed for the tasks performed in this sub-repo. Whatever is installed as a package, make sure to install them directly as a package and do not go for building from the source unless you know how to handle the compatibility issues that may come up.
	
1. RPI OS already comes with a system Python installed. So, you will find that pip and setuptools might already be present.
2. I have used mamba and miniconda in RPI before, they work fine but when it comes to handling the tug of war with numpy in the centre, I believe the native virtual environment of Python is able to better handle it at the moment (when I worked on these tasks, things might change in the future)
3. So, will install some python modules first directly in the host python. Run the following command
   sudo apt install -y python3-picamera2 python3-libcamera python3-opencv python3-numpy python3-matpllotlib python3-skimage libcap-dev
4. Create a system-linked virtual environment 

	python3 -m venv ~/cvpy --system-site-packages
	
 	This venv will use the site-packages from the host python which means the numpy opencv picamera2 libcamera that are present in your host python will be 		available to your venv as well. This prevents any module that you install inside the venv from updating or changing these modules directly during its own 		installation. It’s a clean way to pin certain versions of numpy and opencv in your RPI and protect you from opening  big can of worms. Many python modules 		like torch, tensorflow, huggingface, etc try to change the numpy version according to their needs which many times leads to compatibility issues that pip 		cannot fix.

5. Activate the virtual environment
	source ~/cvpy/bin/activate
		
	
# TESTING THE CAMERA
	
1. Power off your RPI (if it is on) - run "sudo poweroff" and remove the power adapter cable - very important
2. Attach the Camera cable to the 5MP RPI camera
	i. This is a very simple step but I have found that I have made mistakes many times
	ii. Your camera cable for RPI 5 would look something like this

   	<img width="708" height="649" alt="image" src="https://github.com/user-attachments/assets/dbaf7d30-934c-43b6-b65b-a761756f2caa" />
	
 	<img width="818" height="753" alt="image" src="https://github.com/user-attachments/assets/e86e1ebe-40e4-4b69-97cb-92a9eda0d70c" />
	
   	iii. The RPI 5 has a narrower CSI port than the previous versions of RPI. So the narrower of the ends is meant for RPI CSI port and the other one for the 			camera. There is a high chance that the cable that your camera came with is not compatible with the CSI port of the RPI 5. You can refer to the attached 		images to understand exactly what a RPI 5 camera cable looks like.
	iv. The cable (either end) has two sides to it - one will have the copper wire ends (golden colour) and the other side will blacked out or blued out.
	v. The CSI port on your RPI will look like this

   	<img width="529" height="417" alt="image" src="https://github.com/user-attachments/assets/c3d40339-867d-4034-b7a5-17ef6814b114" />
	
    vi. Notice that the CSI port has two side as well - one with the wires and one without the wires. 
	vii. Pull the CSI port attachment cap gently (it will come up slightly)
	viii. While connecting the cable to the CSI port, you need to make sure that the copper wired end of the cable is facing or is in contact with the wired side 		of the CSI port on the RPI
	ix. Make sure that the cable is fully in and then gently press the cap to make the cable fit tightly inside the port.
	x. A CSI port will present on your camera as well with the same type of cap.
	xi. The fatter end of the cable will go inside the CSI port of the camera module. 
	xii. Make sure to put the cable with the copper wired side facing or in contact with the wired end of the CSI port on the camera module as well, just as it 		was done on the RPI board as well.
3. Now we can test the camera module
	i. Reattach the power cable and turn on your RPI
	ii. On a terminal in your RPI (via RealVNC Viewer), run this command
		1) rpicam-hello -t 5000
		2) You should see a window open up that will show you the live feed of your camera
	iii. You can run the following command to take a snapshot with your camera
		1) rpicam-jpeg -o test.jpg
	iv. You can also keep the video feed open indefinitely (or untill you press CTRL+C on the terminal)
		1) rpicam-vid -t 0
	v. The hello and vid command will not work if you are using simple ssh with MobaXterm or powershell, you will need to do it through your RealVNC Viewer
	vi. If this step fails, the issues could one or more of the following - 
		1) YOUR CAMERA CABLE WAS NOT ATTACHED CORRECTLY - 99% chance of this
		2) Your rpicam library may not be working
		3) There is something wrong with your camera module
		4) And finally and the more scarier one, your RPI CSI port is faulty
4. Another test you can perform with your camera if the last test passes
	i. You can check whether your installed picamera2 library works or not and whether your are able to pass the feed captured by picamera2 to opencv object
	ii. You can run this small script
			
			'''
   			from picamera2 import Picamera2
			import cv2
			
			picam2 = Picamera2()
			config = picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
			picam2.configure(config)
			picam2.start()
			
			while True:
			    frame = picam2.capture_array()
			    cv2.imshow("Original", frame)
			    if cv2.waitKey(1) & 0xFF == ord('q'):
			        break
			
			picam2.stop()
			cv2.destroyAllWindows()
			'''
   
	iii. A window will open up. This will be the imshow window of oepncv module showing your video feed. You can close it by pressing 'q'.
	iv. I have understood that on RPI its better to let picamera2 get the raw feed and then pass it to cv2 for all tasks further down the line.
	
	
If all these steps pass, I believe you are done with the setup and can follow along with the computer vision tasks performed by the codes present in this sub-repo.
