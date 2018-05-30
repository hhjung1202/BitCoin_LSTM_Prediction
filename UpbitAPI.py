#!/usr/bin/python2.7
from websocket import create_connection # websocket-client
import requests # requests
from random import randint, choice
import json
from time import time, sleep
from jwt import encode as jwt_encode # PyJWT

class UpbitAPI():
	UserAgent = 'Mozilla/5.0'
	access_key = None
	secret_key = None

	def __init__(self, access_key, secret_key):
		self.access_key = access_key
		self.secret_key = secret_key

	def getAuthorization(self):
		return 'Bearer ' + jwt_encode({"access_key": self.access_key, "nonce": int(round(time() * 1000))}, self.secret_key, algorithm='HS256')

	def getPrice(self, market):
		url = "https://crix-api-endpoint.upbit.com/v1/crix/candles/minutes/1?code=CRIX.UPBIT.%s&count=1" % (market)
		a = requests.get(url, headers={
			'User-Agent': self.UserAgent
		})
		b = eval(a.content)[0]
		return float(b['tradePrice'])

	def getChart_min(self, market, minute = 15, count = 200):
		url = "https://crix-api-endpoint.upbit.com/v1/crix/candles/minutes/%s?code=CRIX.UPBIT.%s&count=%s" % (str(minute), market, str(count))
		a = requests.get(url, headers={
			'User-Agent': self.UserAgent
		})
		b = eval(a.content)
		return b

	def getChart_min_addition(self, market, to, minute = 15, count = 200):
		url = "https://crix-api-endpoint.upbit.com/v1/crix/candles/minutes/%s?code=CRIX.UPBIT.%s&count=%s&to=%s.000Z" % (str(minute), market, str(count), to[:-6])
		a = requests.get(url, headers={
			'User-Agent': self.UserAgent
		})
		b = eval(a.content)
		return b

		
		# to = "2018-02-15T22:36:00"
		# data['candleDateTime'] = "2018-02-15T22:33:00+00:00"


	def getOrders(self, market):
		url = "http://127.0.0.1:9898/CRIX.UPBIT.%s" % (market)
		a = requests.get(url, headers={
			'User-Agent': self.UserAgent
		})
		return json.loads(a.content)

	def Ask(self, market, price, volume):
		url = "https://ccx.upbit.com/api/v1/orders"
		data = {
			"ord_type":"limit",
			"side":"ask",
			"market":market,
			"price":price,
			"volume":volume
		}
		a = requests.post(url, data=json.dumps(data), headers={
			'User-Agent': self.UserAgent,
			'Authorization': self.getAuthorization(),
			"Content-Type": "application/json"
		})
		return a.content

	def Bid(self, market, price, volume):
		URL = "https://ccx.upbit.com/api/v1/orders"
		data = {
			"ord_type":"limit",
			"side":"bid",
			"market":market,
			"price":price,
			"volume":volume
		}
		a = requests.post(URL, data=json.dumps(data), headers={
			'User-Agent': self.UserAgent,
			'Authorization': self.getAuthorization(),
			"Content-Type": "application/json"
		})

		return a.content

	def Assets(self):
		URL = "https://ccx.upbit.com/api/v1/investments/assets"
		a = requests.get(URL, headers={
			'User-Agent': self.UserAgent,
			'Authorization': self.getAuthorization()
		})
		return eval(a.content)

	def Assets_coin(self,coin):
		asset = self.Assets()
		for x in asset:
			if x['currency'] == coin:
				print x
				print float(x['balance'])
				return float(x['balance'])
		return 0

	def Orders(self, market):
		URL = "https://ccx.upbit.com/api/v1/orders?market=%s&state=wait&limit=300&page=1&order_by=desc" % (market)
		a = requests.get(URL, headers={
			'User-Agent': self.UserAgent,
			'Authorization': self.getAuthorization()
		})
		return eval(a.content)

	def Cancel_market(self,market):
		order = self.Orders(market)
		for x in order:
			self.Cancel(x['uuid'])

	def Cancel(self, uuid):
		URL = "https://ccx.upbit.com/api/v1/order?uuid=%s" % (uuid)
		a = requests.delete(URL, headers={
			'User-Agent': self.UserAgent,
			'Authorization': self.getAuthorization()
		})
		return eval(a.content)

	def Cancel_market_refresh(self,market):
		order = self.Orders(market)
		for x in order:
			self.Cancel(x['uuid'])
			sleep(0.9)

	def Cancel_refresh(self, uuid):
		URL = "https://ccx.upbit.com/api/v1/order?uuid=%s" % (uuid)
		a = requests.delete(URL, headers={
			'User-Agent': self.UserAgent,
			'Authorization': self.getAuthorization()
		})
		return eval(a.content)


if __name__ == '__main__':
	a = UpbitAPI('[ACCESS_KEY]', '[SECRET_KEY]')
	print a.getOrders("KRW-BTC")
	for i in range(5):
		t1 = time()
		print a.Assets()
		t2 = time()
		print t2 - t1

		t1 = time()
		print a.Orders("KRW-BTC")
		t2 = time()
		print t2 - t1
