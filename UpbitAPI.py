#!/usr/bin/python2.7
import requests # requests

class UpbitAPI():
	UserAgent = 'Mozilla/5.0'
	access_key = None
	secret_key = None		

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