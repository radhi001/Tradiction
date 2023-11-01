from django.core.mail import EmailMessage
from django.template.context_processors import csrf
from get_all_tickers import get_tickers as gt
from django.core.checks import Error
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
import mysql.connector
from newsapi import NewsApiClient
from pprint import pprint
import yfinance as yf
from .sentimenttwitter import query_twitter
from django.views.decorators.csrf import csrf_protect, csrf_exempt
import datetime
import hashlib
from django.shortcuts import HttpResponseRedirect

# Create your views here.


def registration(request):
    return render(request, 'registration/trader_register.html')


def expertregistration(request):
    return render(request, 'registration/expert_register.html')


def myprofile(request):
    lid = int(request.session.get('lid'))
    conn = mysql.connector.connect(host='localhost', database='tradiction', user='root', password='Sanju@1998')
    cursor = conn.cursor()
    query = "SELECT role from login where lid='%d'" % lid
    cursor.execute(query)
    row = cursor.fetchone()
    role = row[0]
    if role == 'trader':
        query = "SELECT * from traderreg join login l on traderreg.lid = l.lid where l.lid='%d'" % lid
        cursor.execute(query)
        rows = cursor.fetchall()
        return render(request, 'myprofile.html', {'rows': rows, 'lid': lid})

    elif role == 'expert':
        query = "SELECT * from expertreg join login l on expertreg.loid = l.lid where l.lid='%d'" % lid
        cursor.execute(query)
        rows = cursor.fetchall()
        return render(request, 'myprofile.html', {'rows': rows})


def updateinfo(request):
    fname = "\"" + request.POST.get('fname') + "\""
    lname = "\"" + request.POST.get('lname') + "\""
    email = "\"" + request.POST.get('email') + "\""
    address = "\"" + request.POST.get('address') + "\""
    city = "\"" + request.POST.get('city') + "\""
    state = "\"" + request.POST.get('state') + "\""
    phono = "\"" + request.POST.get('phono') + "\""
    lid = int(request.session.get('lid'))
    conn = mysql.connector.connect(host='localhost', database='tradiction', user='root', password='Sanju@1998')
    cursor = conn.cursor()
    query = "UPDATE traderreg SET firstname=%s, lastname=%s, address=%s, city=%s, state=%s, phoneno=%s where lid=%d" %(fname, lname, address, city, state, phono, lid)
    cursor.execute(query)
    conn.commit()
    query = "UPDATE login SET username=%s where lid=%d" %(email, lid)
    cursor.execute(query)
    conn.commit()
    return myprofile(request)


def updatebankinfo(request):
    ssnno = request.POST.get('ssnno')
    routingno = request.POST.get('routingno')
    accno = request.POST.get('accountno')
    lid = int(request.session.get('lid'))
    conn = mysql.connector.connect(host='localhost', database='tradiction', user='root', password='Sanju@1998')
    cursor = conn.cursor()
    query = "UPDATE traderreg SET ssnno=%s, routingno=%s, accountno=%s where lid=%d" % (ssnno, routingno, accno, lid)
    cursor.execute(query)
    conn.commit()
    return myprofile(request)


def updatepwd(request):
    key = "tradiciton"
    pwd = request.POST.get('pwd')
    lid = int(request.session.get('lid'))
    conn = mysql.connector.connect(host='localhost', database='tradiction', user='root', password='Sanju@1998')
    cursor = conn.cursor()
    query = "UPDATE login SET password=%s where lid=%d" %(pwd, lid)
    cursor.execute(query)
    return myprofile(request)


def portfolio(request):
    lid = request.session.get('lid')
    print(lid)
    if lid is not None:
        lid = int(request.session.get('lid'))
        conn = mysql.connector.connect(host='localhost', database='tradiction', user='root', password='Sanju@1998')
        cursor = conn.cursor()
        query = " select bs.bid,bs.datetime,bs.buyprice, bs.quantity,bs.total, sd.stockname, sd.stocksymbol from buystocks as bs join stocks as sd " \
                "on bs.stid = sd.sid where bs.lid = '%d' AND status='payment successful'" % lid
        cursor.execute(query)
        rows = cursor.fetchall()
        print('rows', rows)
        print('length', len(rows))
        #print(rows[0][6])
        sym = []
        for i in rows:
            sym.append(i[6])
        print("sym", sym)
        currentprice = []
        for i in sym:
            symbol = yf.Ticker(i)
            currentprice.append(symbol.info['previousClose'])

        print(currentprice)
        i = 0
        newrows = []
        for j in rows:
            newl = list(j)
            newl.append(currentprice[i])
            newrows.append(newl)
            i = i + 1

        print('updated row', newrows)
        return render(request, 'portfolio.html', {'rows': newrows, 'lid': lid})
    else:
        msg = "You will need to login first"
        return render(request, 'registration/login.html', {'msg': msg})


def sellstocks(request):
    lid = int(request.session.get('lid'))
    bid = int(request.GET.get('bid'))
    print(bid)
    symbol = request.GET.get('symbol')
    print(symbol)
    ticker = yf.Ticker(symbol)
    currentprice = ticker.info['previousClose']
    name = ticker.info['longName']
    conn = mysql.connector.connect(host='localhost', database='tradiction', user='root', password='Sanju@1998')
    cursor = conn.cursor()
    query = "select stid,datetime,buyprice,quantity from buystocks where bid='%d' " % (bid)
    cursor.execute(query)
    rows = cursor.fetchall()
    print(rows)
    return render(request, 'sellstocks.html',
                  {'name': name, 'rows': rows, 'price': currentprice, 'symbol': symbol, 'bid': bid, 'lid': lid})


def sells(request):
    symbol = request.POST.get('symbol')
    lid = int(request.session.get('lid'))
    sname = request.POST.get('sname')
    sellquantity = int(request.POST.get('sellquantity'))
    buyquantity = int(request.POST.get('buyquantity'))
    left = buyquantity-sellquantity
    bprice = float(request.POST.get('cprice'))
    total = float(request.POST.get('total'))

    sellprice = float(request.POST.get('sellprice'))
    date = datetime.datetime.now().strftime("%c")
    total1 = sellquantity * bprice
    net = total - total1
    bid = int(request.POST.get('bid'))
    conn = mysql.connector.connect(host='localhost', database='tradiction', user='root', password='Sanju@1998')
    cursor = conn.cursor()

    query = "INSERT INTO sellstocks (lid,sellingprice,sellingdate,sellquantity,profitloss,bid) VALUES ('%d','%s','%s','%s','%d','%d')" % (
        lid, sellprice, date, sellquantity,net,bid)
    cursor.execute(query)
    conn.commit()

    query = "UPDATE buystocks SET quantity='%d' where bid=%d" % (left, bid)
    cursor.execute(query)
    conn.commit()
    return tradinghistory(request)


def help(request):
    lid = request.session.get('lid')
    if lid is not None:
        lid = int(lid)
        return render(request, 'help.html', {'lid': lid})
    else:
        return render(request, 'help.html')


def login(request):
    return render(request, 'registration/login.html')


def tradinghistory(request):
    lid = request.session.get('lid')
    print(lid)
    if lid is not None:
        lid = int(lid)
        conn = mysql.connector.connect(host='localhost', database='tradiction', user='root', password='Sanju@1998')
        cursor = conn.cursor()
        query = "SELECT sname,sellingdate,sellingprice,profitloss,buyprice,quantity,total FROM tradiction.sellstocks join tradiction.buystocks on " \
                "sellstocks.bid=buystocks.bid join tradiction.stocks on buystocks.stid=stocks.sid"
        cursor.execute(query)
        rows = cursor.fetchall()
        print(rows)
        return render(request, 'tradinghistory.html', {'rows': rows, 'lid': lid})
    else:
        msg = "You will need to login first"
        return render(request, 'registration/login.html', {'msg': msg})


def logout(request):

    lid = request.session.get('lid')
    print(lid)
    del request.session['lid']
    print(request.META['HTTP_REFERER'])
    #return request.META['HTTP_REFERER'] + request
    return stocks(request)


def between(request):
    conn = mysql.connector.connect(host='localhost', database='tradiction', user='root', password='Sanju@1998')
    cursor = conn.cursor()
    lid = int(request.session.get('lid'))
    bid = int(request.session.get('bid'))
    total = request.session.get('total')
    sid = request.session.get('sid')
    query = "SELECT firstname,username,phoneno FROM traderreg join login on traderreg.lid= login.lid where login.lid ='%d' ;" % lid
    print("between", query)
    cursor.execute(query)
    rows = cursor.fetchall()
    print("between rows", rows)
    return render(request, 'between.html', {'rows': rows, 'total': total, 'bid': bid})


def Home(request):
    lid = request.session.get('lid')
    MERCHANT_KEY = "asBQUwV3"
    key="asBQUwV3"
    SALT = "c7GRmpi4Yb"
    PAYU_BASE_URL = "https://sandboxsecure.payu.in/_payment"
    action = ''
    posted={}
    # Merchant Key and Salt provided y the PayU.
    for i in request.POST:
        posted[i]=request.POST[i]
    hash_object = hashlib.sha256(b'randint(0,20)')
    txnid=hash_object.hexdigest()[0:20]
    hashh = ''
    posted['txnid']=txnid
    hashSequence = "key|txnid|amount|productinfo|firstname|email|udf1|udf2|udf3|udf4|udf5|udf6|udf7|udf8|udf9|udf10"
    posted['key']=key
    hash_string=''
    hashVarsSeq=hashSequence.split('|')
    for i in hashVarsSeq:
        try:
            hash_string+=str(posted[i])
        except Exception:
            hash_string+=''
        hash_string+='|'
    hash_string+=SALT
    hashh=hashlib.sha512(hash_string.encode('utf-8')).hexdigest().lower()
    action =PAYU_BASE_URL
    if(posted.get("key")!=None and posted.get("txnid")!=None and posted.get("productinfo")!=None and posted.get("firstname")!=None and posted.get("email")!=None):
        return render(request, 'current_datetime.html', {"lid":lid, "posted":posted,"hashh":hashh,"MERCHANT_KEY":MERCHANT_KEY,
                                                                               "txnid":txnid,"hash_string":hash_string,"action":"https://sandboxsecure.payu.in/_payment" })
    else:
        return render(request, 'current_datetime.html', {"lid":lid, "posted":posted,"hashh":hashh,"MERCHANT_KEY":MERCHANT_KEY,
                                                                               "txnid":txnid,"hash_string":hash_string,"action":"." })


@csrf_protect
@csrf_exempt
def success(request):
    c = {}
    c.update(csrf(request))
    status=request.POST["status"]
    firstname=request.POST["firstname"]
    amount=request.POST["amount"]
    txnid=request.POST["txnid"]
    posted_hash=request.POST["hash"]
    key=request.POST["key"]
    productinfo=request.POST["productinfo"]
    result = productinfo.split(",")
    bid = result[0]
    lid = result[1]
    request.session['lid'] = lid
    e=request.POST["email"]
    date = datetime.datetime.now().strftime("%c")
    salt="GQs7yium"
    try:
        additionalCharges=request.POST["additionalCharges"]
        retHashSeq=additionalCharges+'|'+salt+'|'+status+'|||||||||||'+e+'|'+firstname+'|'+productinfo+'|'+amount+'|'+txnid+'|'+key
    except Exception:
        retHashSeq = salt+'|'+status+'|||||||||||'+e+'|'+firstname+'|'+productinfo+'|'+amount+'|'+txnid+'|'+key
    hashh=hashlib.sha512(retHashSeq.encode('utf-8')).hexdigest().lower()
    print(hashh)
    print(posted_hash)
    if(hashh !=posted_hash):
        print("Invalid Transaction. Please try again")
    else:
        print("Thank You. Your order status is ", status)
        print("Your Transaction ID for this transaction is ",txnid)
        print("We have received a payment of Rs. ", amount ,". Your order will soon be shipped.")
    #return render(request, 'sucess.html', {"txnid":txnid,"status":status,"amount":amount})
    conn = mysql.connector.connect(host='localhost', database='tradiction', user='root', password='Sanju@1998')
    cursor = conn.cursor()
    query = "INSERT INTO payment (bid,tranid,status,date) VALUES('%s','%s','%s','%s')" % (bid, txnid, status, date)
    cursor.execute(query)
    conn.commit()
    query = "UPDATE buystocks SET status='payment successful' where bid=%s" % (bid)
    cursor.execute(query)
    conn.commit()
    subject = "PAYMENT SUCCESSFUL"
    Body = "HELLO " + firstname + ", Congratulations. Your Payment is successful. Happy Trading."
    print(Body)
    print(e)
    email = EmailMessage(subject, Body, to=[e])
    email.send()
    print("sent")

    return portfolio(request)


@csrf_protect
@csrf_exempt
def failure(request):
    c = {}
    c.update(csrf(request))
    status = request.POST["status"]
    firstname = request.POST["firstname"]
    amount = request.POST["amount"]
    txnid = request.POST["txnid"]
    posted_hash = request.POST["hash"]
    key = request.POST["key"]
    productinfo = request.POST["productinfo"]
    result = productinfo.split(",")
    bid = result[0]
    lid = result[1]
    email = request.POST["email"]
    salt = ""
    try:
        additionalCharges=request.POST["additionalCharges"]
        retHashSeq=additionalCharges+'|'+salt+'|'+status+'|||||||||||'+email+'|'+firstname+'|'+productinfo+'|'+amount+'|'+txnid+'|'+key
    except Exception:
        retHashSeq = salt+'|'+status+'|||||||||||'+email+'|'+firstname+'|'+productinfo+'|'+amount+'|'+txnid+'|'+key
    hashh=hashlib.sha512(retHashSeq.encode('utf-8')).hexdigest().lower()
    if(hashh !=posted_hash):
        print("Invalid Transaction. Please try again")
    else:
        print("Thank You. Your order status is ", status)
        print("Your Transaction ID for this transaction is ",txnid)
        print("We have received a payment of Rs. ", amount ,". Your order will soon be shipped.")
    #return render(request, "Failure.html", c)
    conn = mysql.connector.connect(host='localhost', database='tradiction', user='root', password='Sanju@1998')
    cursor = conn.cursor()
    query = "UPDATE buystocks SET status='payment failed' where bid=%s" % (bid)
    cursor.execute(query)
    conn.commit()
    return stocks(request)


def index(request):

    lid = request.session.get('lid')
    if lid is not None:
        lid = int(lid)
    news = NewsApiClient(api_key='17338a8016484433bcd67895a6a6ed95')

    # /v2/top-headlines
    top_headlines = news.get_top_headlines(
        category='business',
        language='en',
        country='us')
    articles = top_headlines['articles']
    titles = []
    for singlearticle in articles:
        titles.append(singlearticle.get('title'))

    sdetails = []
    symbols = ['AAPL', 'NFLX', 'GOOGL', 'AMZN', 'MSFT', 'UBER', 'TWTR', 'TSLA', 'DELL']
    for i in symbols:
        symbol = yf.Ticker(i)
        print(symbol)
        details = symbol.balance_sheet
        print(details)
        sdetails.append(details)
    return render(request, 'home.html', {'articles': articles, 'lid': lid, 'details': sdetails})


"""def search(request):

    try:
        conn = mysql.connector.connect(host='localhost', database='tradiction', user='admin1', password='toor)
        cursor = conn.cursor()
        query = 'select count(*) from stocks'
        cursor.execute(query)
        rows = cursor.fetchall()
        print(rows)
        if rows is None:

            payload = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            table = payload[0]

            symbols = table['Symbol'].values.tolist()
            print(symbols)

            names = table['Security'].values.tolist()
            print(names)

            query = 'TRUNCATE `tradiction`.`stocks`'
            cursor.execute(query)
            for i, j in zip(symbols, names):
                i = "\"" + i + "\""
                j = "\"" + j + "\""
                print(i, j)
                query = 'INSERT INTO stocks (`symbol`, `name`) VALUES (%s, %s)' % (i, j)
                cursor.execute(query)

        string = "\"" + "%" + request.GET.get("search") + "%" + "\""
        print(string)
        query = 'SELECT stocksymbol, stockname from stocks where stockname LIKE %s' % string
        cursor.execute(query)
        rows = cursor.fetchall()
        print(rows)
        return render(request, 'stocks.html', {"row": rows})

    except Error as e:
        print(e)

    finally:
        cursor.close()
        conn.close()
"""

def addtrader(request):

    key = "tradiction123456"
    fname = "\"" + request.POST.get('fname') + "\""
    lname = "\"" + request.POST.get('lname') + "\""
    email = "\"" + request.POST.get('email') + "\""
    address = "\"" + request.POST.get('address') + "\""
    city = "\"" + request.POST.get('city') + "\""
    state = "\"" + request.POST.get('state') + "\""
    phono = "\"" + request.POST.get('phono') + "\""
    ssnno = request.POST.get('ssnno')
    bankst = request.FILES['bankst']
    routingno = request.POST.get('routingno')
    accno = request.POST.get('accno')
    pwd = request.POST.get('cpwd1')
    cpwd = request.POST.get('cpwd2')

    conn = mysql.connector.connect(host="localhost", database="tradiction", user="admin1", password="Admin123")
    cursor = conn.cursor()

    if request.method == 'POST':
        fs = FileSystemStorage()

        # Bank Statement

        filename = bankst.name
        extension = filename.split(".")
        upload_file_name = request.POST.get('fname') + request.POST.get('lname') + "_bankstatement." + extension[1]
        filename = fs.save(upload_file_name, bankst)
        bankstatement_url = "\"" + fs.url(filename) + "\""
        print(bankstatement_url)

    if pwd == cpwd:
        query = "insert into tradiction.login (username,password,role) values (%s,%s,%s)" % (email, pwd, '"trader"')
        cursor.execute(query)

        id = cursor.lastrowid

        query = "insert into tradiction.traderreg (firstname, lastname, address, city, state, phoneno, ssnno, bankst, routingno, accountno, lid, " \
                "status) " \
                "values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%d,%s)" % (
                fname, lname, address, city, state, phono, ssnno, bankstatement_url, routingno, accno, id, '"pending"')

        cursor1 = conn.cursor()
        cursor1.execute(query)
        conn.commit()
        msg = "Registration Complete !! Wait for confirmation message until your account is reviewed"
        return render(request, 'registration/login.html', {'msg': msg})

    else:
        msg = "Passwords do not match"
        return render(request, 'registration/trader_register.html', {'msg': msg})


def addexpert(request):
    fname = "\"" + request.POST.get('fname') + "\""
    lname = "\"" + request.POST.get('lname') + "\""
    email = "\"" + request.POST.get('email') + "\""
    address = "\"" + request.POST.get('address') + "\""
    city = "\"" + request.POST.get('city') + "\""
    state = "\"" + request.POST.get('state') + "\""
    postalcode = "\"" + request.POST.get('postalcode') + "\""
    phono = "\"" + request.POST.get('phono') + "\""
    certificate = request.FILES['certificate']
    ssnno = request.POST.get('ssnno')
    pwd = request.POST.get('cpwd1')
    cpwd = request.POST.get('cpwd2')

    conn = mysql.connector.connect(host="localhost", database="tradiction", user="admin1", password="Admin123")
    cursor = conn.cursor()

    if request.method == 'POST':
        fs = FileSystemStorage()
        filename = certificate.name
        extension = filename.split(".")
        upload_file_name = request.POST.get('fname') + request.POST.get('lname') + "_certificate." + extension[1]
        filename = fs.save(upload_file_name, certificate)
        certificate_url = "\"" + fs.url(filename) + "\""
        print(certificate_url)

    if pwd == cpwd:
        query = "insert into tradiction.login (username,password,role) values (%s,%,%s)" % (email, pwd, '"expert"')
        cursor.execute(query)
        id = cursor.lastrowid

        query = "insert into tradiction.expertreg (firstname, lastname, address, city, state, postalcode, phoneno, ssnno, expertcertificate, loid, " \
                "status" \
                ") " \
                "values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%d,%s)" % (
                    fname, lname, address, city, state, postalcode, phono, ssnno, certificate_url, id, '"pending"')

        cursor.execute(query)
        conn.commit()
        msg = "Registration Complete !! Wait for confirmation message until your account is reviewed"
        return render(request, 'registration/login.html', {'msg': msg})

    else:
        msg = "Passwords do not match"
        return render(request, 'registration/expert_register.html', {'msg': msg})


def forgotpwd(request):
    return render(request, 'forgotpwd.html')


def getpwd(request):
    e = request.POST.get("username")
    print(e)
    conn = mysql.connector.connect(host='localhost', database='tradiction', user='root', password='Sanju@1998')
    cursor = conn.cursor()
    query = "select password from login where username = '%s'" % (e)
    cursor.execute(query)
    rows = cursor.fetchone()
    print(rows)
    pwd = rows[0]
    subject = "FORGOT PASSWORD"
    Body = "HELLO THERE, your password is " + pwd
    print(Body)
    to = e
    try:
        # em = EmailMessage(subject, Body, ['miranisanjay417@gmail.com'])
        # em.send()
        email = EmailMessage(subject, Body, to=[e])
        email.send()

        print("sent")
    except Error as e:
        print(e)

    return login(request)


def logindata(request):

    uname = request.POST.get('username')
    pwd = request.POST.get('password')

    conn = mysql.connector.connect(host='localhost', database='tradiction', user='root', password='Sanju@1998')
    cursor = conn.cursor()
    query = "SELECT password,lid,role FROM login WHERE username= '%s'" % uname
    cursor.execute(query)
    rows = cursor.fetchall()
    print(rows)

    if rows:
        loginpwd = rows[0][0]
        loginid = rows[0][1]
        loginrole = rows[0][2]

        request.session['lid'] = loginid

        if pwd == loginpwd:
            if loginrole == 'admin':
                return render(request, 'admin.html', {'lid': loginid})

            elif loginrole == 'expert':
                query = "SELECT status FROM expertreg WHERE loid ='%d' " % (loginid)
                cursor.execute(query)
                rows = cursor.fetchall()
                status = rows[0][0]

                if status == 'Accepted':
                    return index(request)

                elif status == 'Rejected':
                    msg = " Sorry, Your Registration As an Expert has been Reject  :(  "
                    return render(request, 'registration/login.html', {'msg': msg})

                else:
                    msg = " Wait For Confirmation Message !!!  "
                    return render(request, 'registration/login.html', {'msg': msg})

            elif loginrole == 'trader':
                query = "SELECT status FROM traderreg WHERE lid ='%d' " % loginid
                cursor.execute(query)
                rows = cursor.fetchone()
                print(rows)
                status = rows[0]
                print(status)

                if status == 'Accepted':
                    return stocks(request)

                elif status == 'Rejected':
                    msg = " Sorry, Your Registration As a Traders has been Reject  :(  "
                    return render(request, 'registration/login.html', {'msg': msg})

                else:
                    msg = " Wait For Confirmation Message !!!  "
                    return render(request, 'registration/login.html', {'msg': msg})

        else:
            msg = "Entered Password is incorrect!!!"
            return render(request, "registration/login.html", {'msg': msg})

    else:
        msg = "Given username does not exist!!!"
        return render(request, "registration/login.html", {'msg': msg})


def verifyexpert(request):
    try:
        key = "tradiction123456"
        lid = int(request.session.get('lid'))
        conn = mysql.connector.connect(host='localhost', database='tradiction', user='root', password='Sanju@1998')
        cursor = conn.cursor()
        query = " SELECT * FROM expertreg WHERE status ='pending'"
        cursor.execute(query)
        rows = cursor.fetchall()

        return render(request, 'expertverification.html', {'rows': rows, 'lid': lid})

    except Error as e:
        print(e)

    finally:
        cursor.close()
        conn.close()


def verifytrader(request):
    try:
        key = "tradiction123456"
        lid = int(request.session.get('lid'))
        conn = mysql.connector.connect(host='localhost', database='tradiction', user='root', password='Sanju@1998')
        cursor = conn.cursor()
        query = " SELECT * FROM traderreg WHERE status ='pending'"
        cursor.execute(query)
        rows = cursor.fetchall()
        return render(request, 'traderverification.html', {'rows': rows, 'lid': lid})

    except Error as e:
        print(e)

    finally:
        cursor.close()
        conn.close()


def accepttrader(request):
    try:
        id = int(request.GET.get('id'))
        conn = mysql.connector.connect(host='localhost', database='tradiction', user='root', password='Sanju@1998')
        cursor = conn.cursor()
        query = " update traderreg SET status='Accepted' WHERE tid = '%d'" % id
        cursor.execute(query)
        conn.commit()
        query = "SELECT username from login join traderreg on login.lid = traderreg.lid where traderreg.tid = %d" %id
        cursor.execute(query)
        rows = cursor.fetchall()
        print(rows)
        e = rows[0][0]
        subject = "Welcome to Tradiction"
        Body = "Congratulations, Your account is now approved"
        email = EmailMessage(subject, Body, to=[e])
        email.send()
        print("sent")
        return verifytrader(request)

    except Error as e:
        print(e)

    finally:
        cursor.close()
        conn.close()


def rejecttrader(request):
    try:
        id = int(request.GET.get('id'))
        conn = mysql.connector.connect(host='localhost', database='tradiction', user='root', password='Sanju@1998')
        cursor = conn.cursor()
        query = " update traderreg SET status='Rejected' WHERE tid  = '%d'" % id
        cursor.execute(query)
        conn.commit()
        query = "SELECT username from login join traderreg on login.lid = traderreg.lid where traderreg.tid = %d" % id
        cursor.execute(query)
        rows = cursor.fetchall()
        print(rows)
        e = rows[0][0]
        subject = "Account Rejected"
        Body = "We appreciate your patience but your account has been rejected."
        email = EmailMessage(subject, Body, to=[e])
        email.send()
        print("sent")
        return verifytrader(request)

    except Error as e:
        print(e)

    finally:
        cursor.close()
        conn.close()


def acceptexpert(request):
    try:
        id = int(request.GET.get('id'))
        conn = mysql.connector.connect(host='localhost', database='tradiction', user='root', password='Sanju@1998')
        cursor = conn.cursor()
        query = " update expertreg SET status='Accepted' WHERE eid = '%d'" % id
        cursor.execute(query)
        conn.commit()
        query = "SELECT username from login join expertreg e on login.lid = e.loid where e.eid = %d" % id
        cursor.execute(query)
        rows = cursor.fetchall()
        print(rows)
        e = rows[0][0]
        subject = "Welcome to Tradiction"
        Body = "Congratulations, Your account is now approved"
        email = EmailMessage(subject, Body, to=[e])
        email.send()
        print("sent")
        return verifyexpert(request)

    except Error as e:
        print(e)

    finally:
        cursor.close()
        conn.close()


def rejectexpert(request):
    try:
        id = int(request.GET.get('id'))
        conn = mysql.connector.connect(host='localhost', database='tradiction', user='root', password='Sanju@1998')
        cursor = conn.cursor()
        query = " update expertreg SET status='Rejected' WHERE eid = '%d'" % id
        cursor.execute(query)
        conn.commit()
        query = "SELECT username from login join expertreg e on login.lid = e.loid where e.eid = %d" % id
        cursor.execute(query)
        rows = cursor.fetchall()
        print(rows)
        e = rows[0][0]
        subject = "Account Rejected"
        Body = "We appreciate your patience but your account has been rejected."
        email = EmailMessage(subject, Body, to=[e])
        email.send()
        print("sent")
        return verifyexpert(request)

    except Error as e:
        print(e)

    finally:
        cursor.close()
        conn.close()


def news(request):

    lid = request.session.get('lid')
    topnews = NewsApiClient(api_key='17338a8016484433bcd67895a6a6ed95')

    # /v2/top-headlines
    top_headlines = topnews.get_everything(
        q='market',
        language='en')
    articles = top_headlines['articles']

    top_headlines = topnews.get_everything(
        q='stock',
        language='en')
    articles.extend(top_headlines['articles'])

    top_headlines = topnews.get_top_headlines(
                      category='business',
                      language='en',
                      country='us')

    articles.extend(top_headlines['articles'])
    return render(request, 'news.html', {'articles': articles, 'lid': lid})


def loadstocks(request):
    list_of_tickers = gt.get_tickers()
    pprint(list_of_tickers)
    name = []
    for i in list_of_tickers:
        symbol = yf.Ticker(i)
        name.append(symbol.info['longName'])

    pprint(name)


def stocks(request):
    lid = request.session.get('lid')
    print(lid)
    conn = mysql.connector.connect(host='localhost', database='tradiction', user='root', password='Sanju@1998')
    cursor = conn.cursor()
    if lid is not None:
        lid = int(lid)
        query = "SELECT symbol FROM watchlist where logid = %d" % lid
        cursor.execute(query)
        rows1 = cursor.fetchall()
        print(rows1)
        symbols = []
        for i in rows1:
            symbols.append(i[0])

    query = "SELECT * FROM stocks"
    cursor.execute(query)
    rows = cursor.fetchall()

    if lid:
        return render(request, 'stocks.html', {'rows': rows, 'symbols': symbols, 'lid': lid})
    else:
        return render(request, 'stocks.html', {'rows': rows})


def stockdetails(request):

    lid = request.session.get('lid')
    conn = mysql.connector.connect(host='localhost', database='tradiction', user='root', password='Sanju@1998')
    cursor = conn.cursor()

    ticker = request.GET.get("stock")
    symbol = yf.Ticker(ticker)
    details = symbol.info
    pprint(details)
    ans = query_twitter(details['longName'], 100)

    if lid is not None:
        lid = int(lid)
        query = "SELECT wid FROM watchlist where symbol='%s' and logid = '%d'" % (ticker, lid)
        cursor.execute(query)
        rows = cursor.fetchall()
        query = "SELECT sid from stocks where stocksymbol='%s'" % ticker
        cursor.execute(query)
        sid = cursor.fetchone()
        if rows:
            flag = 1
        else:
            flag = 0
        return render(request, 'details.html', {'details': details, 'twitter': ans, 'flag': flag, 'sid': sid, 'lid': lid})

    else:
        return render(request, 'details.html', {'details': details, 'twitter': ans, 'lid': lid})


def addtowatchlist(request):

    lid = request.session.get('lid')
    print(lid)
    if lid is not None:
        lid = int(lid)
        symbol = request.GET.get('symbol')
        print(symbol)
        conn = mysql.connector.connect(host='localhost', database='tradiction', user='root', password='Sanju@1998')
        cursor = conn.cursor()
        query = "INSERT INTO watchlist (logid, symbol) values ('%d','%s') on duplicate " \
                "key update wid=LAST_INSERT_ID(wid=0)" % (lid, symbol)
        cursor.execute(query)
        conn.commit()
        pprint(request.META)
        return HttpResponseRedirect(request.META['HTTP_REFERER'])
    else:
        msg = "You will need to login first"
        return render(request, 'registration/login.html', {'msg': msg})


def watchlist(request):
    lid = request.session.get('lid')
    if lid is not None:
        lid = int(lid)
        conn = mysql.connector.connect(host='localhost', database='tradiction', user='root', password='Sanju@1998')
        cursor = conn.cursor()
        query = "SELECT wid,logid,stockname,stocksymbol FROM watchlist join stocks on watchlist.symbol = stocks.stocksymbol where watchlist.logid = %d" % lid
        cursor.execute(query)
        rows = cursor.fetchall()
        return render(request, 'watchlist.html', {'rows': rows, 'lid': lid})
    else:
        msg = "You will need to login first"
        return render(request, 'registration/login.html', {'msg': msg})


def removefromwatchlist(request):

    lid = int(request.session.get('lid'))
    symbol = request.GET.get('symbol')
    print(symbol)
    conn = mysql.connector.connect(host='localhost', database='tradiction', user='root', password='Sanju@1998')
    cursor = conn.cursor()
    query = "DELETE FROM watchlist where symbol='%s' and logid=%d" % (symbol, lid)
    cursor.execute(query)
    conn.commit()
    return HttpResponseRedirect(request.META['HTTP_REFERER'])


def addstocks(request):

    sid = int(request.GET.get('id'))
    ticker = request.GET.get('symbol')
    lid = int(request.session.get('lid'))
    symbol = yf.Ticker(ticker)
    details = symbol.info
    pprint(details)
    buyprice = round(details['previousClose'], 3)
    name = details['longName']
    details = [name, buyprice, sid]
    return render(request, 'buystocks.html', {'details': details, 'lid': lid})


def buystocks(request):
    sid = request.POST.get('sid')
    print(sid)
    cprice = request.POST.get('cprice')
    sname = request.POST.get('sname')
    quantity = request.POST.get('quantity')
    total = request.POST.get('total')
    lid = int(request.session.get('lid'))
    date = datetime.datetime.now().strftime("%c")
    request.session['sid'] = sid
    request.session['total'] = total
    conn = mysql.connector.connect(host='localhost', database='tradiction', user='root', password='Sanju@1998')
    cursor = conn.cursor()

    query = "insert into buystocks(lid,stid,sname,datetime,buyprice,quantity,total,status) values ('%d','%s','%s','%s','%s','%s','%s','%s')" \
            % (lid, sid, sname, date, cprice, quantity, total, "pending")
    print("add stock", query)
    cursor.execute(query)
    conn.commit()
    bid = cursor.lastrowid
    request.session['bid'] = bid
    return between(request)
