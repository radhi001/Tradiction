#1. Create a new file to store whitelisted users (correctly registered user format: username: password), and then create a file to store blacklisted users (users who entered the wrong username three times).
#2. Read the whitelist file, assign the content to a variable, and close it.
#3. Divide the variable with ":", the first digit (index 0) is assigned to username, and the second digit (index 1) is assigned to password.
#4. Read the blacklist file, assign the content to a variable, and close it.
#5. Define a variable (t) to store the number of user inputs,
#6. Cycling, when the number of cycles is less than three times, continue to cycle, when the number of times is greater than three times, it will prompt you to enter more than three times, the account is locked,
#During recycling, enter the user name, and determine whether the user name is in the blacklist in a loop. If it is, then it prompts "This account has been locked." Medium, continue to judge whether the password is correct.

# -*- coding:utf-8 -*-
#Need to simulate user login, more than three error locks are not allowed to log in
  
  
count = 0
  
#realname passwd
Real_Username = "test"
Real_Password = "test"
  
#Read the contents of the blacklist
f = open('black_user','r')
lock_file = f.read()
f.close()
  
Username = input('Enter your username:')
  
#Determine whether the input user is in the blacklist, if it is, it is not allowed to enter the password
for i in range(1):
 if lock_file == Username:
 print('Sorry, your user is locked, temporarily not allowed to use!')
 exit()
 else:
 continue
  
#Try to enter the password and count the number of entries
for i in range(3):
 Password = input("Enter your password:")
 if Username == Real_Username and Password == Real_Password:
 print("login successful")
 break
 else:
 print("login failed")
 count += 1
  
#If you enter the wrong password three times, you will be prompted to lock the username and black out the username
if count == 3:
 print("Sorry, the number of incorrect passwords you entered has reached 3 times, and your username will be locked")
 f = open("black_user","w")
 f.write("%s"%Username)
 f.close()
