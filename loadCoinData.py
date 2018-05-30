# version - python 2 

import UpbitAPI
import numpy as np
import math

class Calculate:

	def __init__(self, api, target_coin):

		self.api = api
		self.target_coin = target_coin
		
		self.rsi_min = []
		self.rsi_15 = []
		
		self.data = []
		self.data_15 = []

		self.data = api.getChart_min('KRW-BTC')

		self.ma = []
		self.boll = []

	def extendable_data(self):
		
		for j in range(1,60):
			temp_add = self.api.getChart_min_addition('KRW-' + self.target_coin, self.data[0]['candleDateTime'])
			
			temp_add.reverse()
			for i in self.data:
				temp_add.append(i)
			self.data = temp_add

		print self.data[0]['candleDateTime']
		print self.data[-1]['candleDateTime']


	def rsiFunc(self, source, n=5):
	
		data_edit = []
		for each in source:
			data_edit.append(each['tradePrice'])

		deltas = np.diff(data_edit)
		seed = deltas[:n+1]
		up = seed[seed>=0].sum()/n
		down = -seed[seed<0].sum()/n
		rs = up/down
		rsi = np.zeros_like(data_edit)
		rsi[:n] = 100. - 100./(1.+rs)

		for i in range(n, len(data_edit)):
			delta = deltas[i-1] # cause the diff is 1 shorter

			if delta>0:
				upval = delta
				downval = 0.
			else:
				upval = 0.
				downval = -delta

			up = (up*(n-1) + upval)/n
			down = (down*(n-1) + downval)/n

			rs = up/down
			rsi[i] = 100. - 100./(1.+rs)

		return rsi
	def MovingAverageFunc(self, source, n=15):
		data_edit = []
		for each in source:
			data_edit.append(each['tradePrice'])

		ma = np.zeros_like(data_edit)
		for i in range(0, n): # 0 ~ 49
			ma[i] = float(sum(data_edit[:i+1])) / (i+1)

		for i in range(n, len(data_edit)): # 50 ~199 
			ma[i] = sum(data_edit[i-n+1:i+1]) / n

		return ma
	def BollingerFunc(self, source, n=15):
		data_edit = []
		for each in source:
			data_edit.append(each['tradePrice'])

		# diff = np.subtract(data_edit,ma)
		boll = np.zeros_like(data_edit)
		multiplier = 2.0

		for i in range(n, len(data_edit)):
			boll[i] = multiplier*self.tostand(data_edit[(i-n):i], self.ma[i])

		return boll
	def tostand(self, seq, ma, n= 15):
		stand = 0.0
		sum1 = 0.0
		for x in seq:
			sum1 = sum1 + (ma-float(x))**2
		stand = math.sqrt(sum1/n)
		return stand

	def SET_RSI(self):
		self.extendable_data()
		self.rsi_min = self.rsiFunc(self.data)
		self.ma = self.MovingAverageFunc(self.data)
		self.boll = self.BollingerFunc(self.data)

	def printALL(self):
		print 'data', len(self.data)
		print self.data[-1]
		print 'ma', len(self.ma)
		print self.ma[-1]
		print self.ma[0]
		print 'boll', len(self.boll)
		print 'rsi', len(self.rsi_min)
		print self.rsi_min[-1]
		print self.rsi_min[0]

	def writeFile(self):
		f = open("data.csv", 'w')
		write_data = '%s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n' % (
			'candleDateTimeKst', 
			'candleAccTradePrice', 
			'candleAccTradeVolume', 
			'openingPrice', 
			'lowPrice', 
			'highPrice', 
			'tradePrice', 
			'ma', 
			'boll', 
			'rsi')
		
		f.write(write_data)
		for i in range(0,len(self.data)):
			write_data = '%s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n' % (
				str(self.data[i]['candleDateTimeKst']),
				str(self.data[i]['candleAccTradePrice']),
				str(self.data[i]['candleAccTradeVolume']),
				str(self.data[i]['openingPrice']),
				str(self.data[i]['lowPrice']),
				str(self.data[i]['highPrice']),
				str(self.data[i]['tradePrice']),
				str(self.ma[i]),
				str(self.boll[i]),
				str(self.rsi_min[i]))
			f.write(write_data)

		f.close()

	def start(self):
		self.SET_RSI()
		self.printALL()
		self.writeFile()

if __name__ == '__main__':

	api = UpbitAPI.UpbitAPI()

	target_coin = "BTC"
	Calc_Class = Calculate(api, target_coin)
	Calc_Class.start()
