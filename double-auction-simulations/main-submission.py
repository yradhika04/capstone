#!python3 

"""
Simulation of single-type multi-unit double-auction mechanisms.

Author: Erel Segal-Halevi
Since : 2017-07
"""

import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import plotting
import matplotlib.pyplot as plt
import math
import os
import random

from doubleauction import MUDA,WALRAS,walrasianEquilibrium,randomTradeWithExogeneousPrice
import torq_datasets_read as torq
from random_datasets import randomAuctions

COLUMNS=(
	'Total buyers', 'Total sellers', 'Total traders', 'Min total traders', 'Total units', 
	'Max units per trader', 'Min units per trader', 'Normalized max units per trader', 'stddev',
	'Optimal buyers', 'Optimal sellers', 'Optimal units',
	'Optimal gain', 'MUDA-lottery gain', 'MUDA-Vickrey traders gain', 'MUDA-Vickrey total gain')

def replicaAuctions(replicaNums:list, auctions:list):
	"""
	INPUT: auctions - list of m auctions;  
	       replicaNums - list of n integers.
	OUTPUT: generator of m*n auctions, where in each auction, each agent is replicated i times.
	"""
	for auctionID,auctionTraders in auctions:
		for replicas in replicaNums:
			traders = replicas * auctionTraders 
			yield auctionID,traders

def sampleAuctions(agentNums:list, auctions:list):
	"""
	INPUT: auctions - list of m auctions;  
	       agentNums - list of n integers.
	OUTPUT: generator of m*n auctions, where in each auction, i agents are sampled from the empirical distribution
	"""
	for auctionID,auctionTraders in auctions:
		for agentNum in agentNums:
			traders = [random.choice(auctionTraders) for i in range(agentNum)]
			yield auctionID,traders


def simulateAuctions(auctions:list, resultsFilename:str, keyColumns:list):
	"""
	Simulate the auctions in the given generator.
	"""
	col1 = ("Lottery Winning Buyers", "Lottery Winning Sellers")
	col2 = ("Vickrey Winning Buyers","Vickrey Winning Sellers")
	col3 = ("Lottery Time taken")
	col4 = ("Vickrey Time taken")
	columns = keyColumns+COLUMNS
	results = DataFrame(columns=columns)
	traderLotteryResults = DataFrame(columns=col1)		
	traderVickreyResults = DataFrame(columns=col2)	
	timeTakenLotteryResults = DataFrame(columns=[col3])		
	timeTakenVickreyResults = DataFrame(columns=[col4])	
	print("\t{}".format(columns))

	resultsFilenameTemp = resultsFilename+".temp"
	traderLotteryFileName = "results/tradersLotteryResults.csv"
	traderVickreyFileName = "results/tradersVickreyResults.csv"
	traderLotteryFileNameTemp = "results/tradersLottery.csv.temp"
	traderVickreyFileNameTemp = "results/tradersVickrey.csv.temp"
	timeLotteryFileName = "results/timeTakenLotteryResults.csv"
	timeVickreyFileName = "results/timeTakenVickreyResults.csv"
	timeLotteryFileNameTemp = "results/timeTakenLottery.csv.temp"
	timeVickreyFileNameTemp = "results/timeTakenVickrey.csv.temp"
	for auctionID,traders in auctions:
		flag1 = False
		flag2 = False
		if not traders:
			raise ValueError("traders for auction {} is empty", auctionID)
		print("Simulating auction {} with {} traders".format(auctionID,len(traders)))
		totalBuyers = sum([t.isBuyer for t in traders])
		totalSellers = len(traders)-totalBuyers
		unitsPerTrader = [t.totalUnits() for t in traders]
		maxUnitsPerTrader = max(unitsPerTrader)
		minUnitsPerTrader = min(unitsPerTrader)
		stddev = np.sqrt(sum([t.totalUnits()**2 for t in traders]))
		(buyersWALRAS, sellersWALRAS, sizeWALRAS, gainWALRAS) = WALRAS(traders)
		(sizeMUDALottery, gainMUDALottery, gainMUDALottery,TraderIndex1, LotteryTime, sizeMUDAVickrey, tradersGainMUDAVickrey, totalGainMUDAVickrey,TraderIndex2, VickreyTime) = MUDA(traders, Lottery=True, Vickrey=True)
		# lotteryBuyerIndex = ""
		# lotterySellerIndex = ""
		# VickreyBuyerIndex = ""
		# VickreySellerIndex = ""
		resultsRow = [
			*auctionID,
			totalBuyers, totalSellers, totalBuyers+totalSellers, min(totalBuyers,totalSellers), sum(unitsPerTrader),
			maxUnitsPerTrader, minUnitsPerTrader, maxUnitsPerTrader/max(1,minUnitsPerTrader), stddev,
			buyersWALRAS, sellersWALRAS, sizeWALRAS,
			gainWALRAS, gainMUDALottery, tradersGainMUDAVickrey, totalGainMUDAVickrey]
		if(len(list(TraderIndex1.values()))>0):
			lotteryBuyerIndex = ",".join(str(i) for i in list(TraderIndex1.values())[0])
			lotterySellerIndex = ",".join(str(i) for i in list(TraderIndex1.values())[1])
			traderLotteryRow = [
				lotteryBuyerIndex,lotterySellerIndex
			]
			timeTakenLotteryRow = [LotteryTime]	
			if(len(lotteryBuyerIndex)>0 and len(lotterySellerIndex)>0):	
				flag1 = True

		if(len(list(TraderIndex2.values()))>0):
			VickreyBuyerIndex = ",".join(str(i) for i in list(TraderIndex2.values())[0])
			VickreySellerIndex = ",".join(str(i) for i in list(TraderIndex2.values())[1])
			traderVickreyRow = [
				VickreyBuyerIndex,VickreySellerIndex
			]
			timeTakenVickreyRow = [VickreyTime]	
			if(len(VickreyBuyerIndex)>0 and len(VickreySellerIndex)>0):		
				flag2 = True

		print("Result Index")
		print("Lengths",len(list(TraderIndex1.values())),len(list(TraderIndex2.values())))
		print(lotteryBuyerIndex)
		print(lotterySellerIndex)
		print(VickreyBuyerIndex)
		print(VickreySellerIndex)
		print("Flags",flag1,flag2)
		print("\t{}".format(resultsRow))
		print("Lottery \t{}".format(traderLotteryRow))
		print("Vickrey \t{}".format(traderVickreyRow))
		results.loc[len(results)] = resultsRow
		if(flag1):
			traderLotteryResults.loc[len(traderLotteryResults)] = traderLotteryRow
			timeTakenLotteryResults.loc[len(timeTakenLotteryResults)] = timeTakenLotteryRow
			traderLotteryResults.to_csv(traderLotteryFileNameTemp)
			timeTakenLotteryResults.to_csv(timeLotteryFileNameTemp)
		if(flag2):
			traderVickreyResults.loc[len(traderVickreyResults)] = traderVickreyRow
			timeTakenVickreyResults.loc[len(timeTakenVickreyResults)] = timeTakenVickreyRow
			traderVickreyResults.to_csv(traderVickreyFileNameTemp)
			timeTakenVickreyResults.to_csv(timeVickreyFileNameTemp)
		results.to_csv(resultsFilenameTemp)
	results.to_csv(resultsFilename)
	traderLotteryResults.to_csv(traderLotteryFileName,index=False)
	traderVickreyResults.to_csv(traderVickreyFileName,index=False)
	timeTakenLotteryResults.to_csv(timeLotteryFileName,index = False)
	timeTakenVickreyResults.to_csv(timeVickreyFileName,index = False)
	os.remove(resultsFilenameTemp)
	os.remove(traderLotteryFileNameTemp)
	os.remove(traderVickreyFileNameTemp)
	os.remove(timeLotteryFileNameTemp)
	os.remove(timeVickreyFileNameTemp)
	return results

def torqSimulationBySymbolDate(filename, combineByOrderDate=False, replicaNums=[1]):
	"""
	Treat each (symbol,date) combination as a separate auction.
	"""
	datasetFilename = "datasets/"+filename+".CSV"
	resultsFilename = "results/"+filename+("-combined" if combineByOrderDate else "")+"-x"+str(max(replicaNums))+".csv" 
	return simulateAuctions(replicaAuctions(replicaNums,
		torq.auctionsBySymbolDate(datasetFilename, combineByOrderDate)),
		resultsFilename, keyColumns=("symbol","date"))

def torqSimulateBySymbol(filename, combineByOrderDate=False, agentNums=[100]):
	"""
	Treat all bidders for the same symbol, in ALL dates, as a distribution of values for that symbol.
	"""
	datasetFilename = "datasets/"+filename+".CSV"
	resultsFilename = "results/"+filename+("-combined" if combineByOrderDate else "")+"-s"+str(max(agentNums))+".csv" 
	return simulateAuctions(sampleAuctions(agentNums,
		torq.auctionsBySymbol(datasetFilename, combineByOrderDate)),
		resultsFilename, keyColumns=("symbol",))


### PLOTS ###

YLABEL = 'Social Welfare Ratio'
YLIM   = [0,1.05]

titleFontSize = 20
legendFontSize = 20
axesFontSize = 20
markerSize = 14


def plotTorq(filename, resultsFilename=None, combineByOrderDate=False, replicaNums=None, agentNums=None, numOfBins=10, ax=None, title=None, xColumn = 'Optimal units'):
	if resultsFilename:
		pass
	elif replicaNums:
		resultsFilename = "results/"+\
			filename+\
			("-combined" if combineByOrderDate else "")+\
			"-x"+str(max(replicaNums))+\
			".csv"
	elif agentNums:
		resultsFilename = "results/"+\
			filename+\
			("-combined" if combineByOrderDate else "")+\
			"-s"+str(max(agentNums))+\
			".csv"
	else:
		raise(Error("cannot calculate resultsFilename"))
	print(resultsFilename)
	plotResults(resultsFilename, xColumn=xColumn, numOfBins=numOfBins, ax=ax, title=title)


def plotResults(resultsFilename=None, xColumn='Min total traders', numOfBins=10, ax=None, title=None):
	if not ax:
		ax = plt.subplot(1, 1, 1)
	if not title:
		title = resultsFilename
	# print("plotting",resultsFilename)

	results = pd.read_csv(resultsFilename)
	results['Optimal market size'] = (results['Optimal buyers']+results['Optimal sellers']) / 2
	results['Normalized market size'] = results['Optimal units'] / (results['Max units per trader'])
	results['log10(M)'] = np.log(results['Max units per trader'])/np.log(10)
	results['MG Satisfaction Ratio' ] = (results['Optimal buyers']+results['Optimal sellers']) / (results['Total traders'])
	print(len(results), " auctions")
	results = results[results['Optimal gain']>0]
	print(len(results), " auctions with positive optimal gain")
	
	
	for field in ['MUDA-lottery', 'MUDA-Vickrey traders', 'MUDA-Vickrey total']:
		results[field+' ratio'] = results[field+' gain'] / results['Optimal gain']

	if numOfBins:
		results_bins = results.groupby(pd.cut(results[xColumn],numOfBins)).mean()
		print('$$$$')
	else:
		results_bins = results.groupby(results[xColumn]).mean()
		print('****')

	results_bins.plot(x=xColumn, y='MUDA-Vickrey total ratio', style=['b^-'], ax=ax, markersize=markerSize)
	results_bins.plot(x=xColumn, y='MUDA-Vickrey traders ratio', style=['gv-'], ax=ax, markersize=markerSize)
	# results_bins.plot(x=xColumn, y='MUDA-lottery ratio', style=['ro-'], ax=ax, markersize=markerSize)

	plt.legend(loc=0,prop={'size':legendFontSize})
	# ax.legend_.remove()
	ax.set_title(title, fontsize= titleFontSize, weight='bold')
	ax.set_ylabel(YLABEL, fontsize= axesFontSize)
	ax.tick_params(axis='both', which='major', labelsize=axesFontSize)
	ax.tick_params(axis='both', which='minor', labelsize=axesFontSize)
	ax.set_ylim(YLIM)

def plotLotteryTorq(filename, resultsFilename=None, combineByOrderDate=False, replicaNums=None, agentNums=None, numOfBins=10, ax=None, title=None, xColumn = 'Optimal units'):
	if resultsFilename:
		pass
	elif replicaNums:
		resultsFilename = "results/"+\
			filename+\
			("-combined" if combineByOrderDate else "")+\
			"-x"+str(max(replicaNums))+\
			".csv"
	elif agentNums:
		resultsFilename = "results/"+\
			filename+\
			("-combined" if combineByOrderDate else "")+\
			"-s"+str(max(agentNums))+\
			".csv"
	else:
		raise(Error("cannot calculate resultsFilename"))
	plotLotteryResults(resultsFilename, xColumn=xColumn, numOfBins=numOfBins, ax=ax, title=title)


def plotLotteryResults(resultsFilename=None, xColumn='Min total traders', numOfBins=10, ax=None, title=None):
	if not ax:
		ax = plt.subplot(1, 1, 1)
	if not title:
		title = resultsFilename
	# print("plotting",resultsFilename)

	results = pd.read_csv(resultsFilename)
	results['Optimal market size'] = (results['Optimal buyers']+results['Optimal sellers']) / 2
	results['Normalized market size'] = results['Optimal units'] / (results['Max units per trader'])
	results['log10(M)'] = np.log(results['Max units per trader'])/np.log(10)
	results['MG Satisfaction Ratio' ] = (results['Optimal buyers']+results['Optimal sellers']) / (results['Total traders'])
	print(len(results), " auctions")
	results = results[results['Optimal gain']>0]
	print(len(results), " auctions with positive optimal gain")
	
	
	for field in ['MUDA-lottery', 'MUDA-Vickrey traders', 'MUDA-Vickrey total']:
		results[field+' ratio'] = results[field+' gain'] / results['Optimal gain']

	if numOfBins:
		results_bins = results.groupby(pd.cut(results[xColumn],numOfBins)).mean()
		print('$$$$')
	else:
		results_bins = results.groupby(results[xColumn]).mean()
		print('****')

	# results_bins.plot(x=xColumn, y='MUDA-Vickrey total ratio', style=['b^-'], ax=ax, markersize=markerSize)
	# results_bins.plot(x=xColumn, y='MUDA-Vickrey traders ratio', style=['gv-'], ax=ax, markersize=markerSize)
	results_bins.plot(x=xColumn, y='MUDA-lottery ratio', style=['b^-'], ax=ax, markersize=markerSize)

	plt.legend(loc=0,prop={'size':legendFontSize})
	# ax.legend_.remove()
	ax.set_title(title, fontsize= titleFontSize, weight='bold')
	ax.set_ylabel(YLABEL, fontsize= axesFontSize)
	ax.tick_params(axis='both', which='major', labelsize=axesFontSize)
	ax.tick_params(axis='both', which='minor', labelsize=axesFontSize)
	ax.set_ylim(YLIM)

def plotSatisfactionFile(filename, resultsFilename=None, combineByOrderDate=False, replicaNums=None, agentNums=None, numOfBins=10, ax=None, title=None, xColumn = 'Optimal units'):
	if resultsFilename:
		pass
	elif replicaNums:
		resultsFilename = "results/"+\
			filename+\
			("-combined" if combineByOrderDate else "")+\
			"-x"+str(max(replicaNums))+\
			".csv"
	elif agentNums:
		print("THERE")
		resultsFilename = "results/"+\
			filename+\
			("-combined" if combineByOrderDate else "")+\
			"-s"+str(max(agentNums))+\
			".csv"
	else:
		print("HERE")
		raise(Error("cannot calculate resultsFilename"))
	print(resultsFilename)

	plotSatisfaction(resultsFilename, xColumn=xColumn, numOfBins=numOfBins, ax=ax, title=title)

def plotSatisfaction(fileName,resultsFilename=None, xColumn='Min total traders', numOfBins=10, ax=None, title=None):
	if not ax:
		ax = plt.subplot(1, 1, 1)
	if not title:
		title = resultsFilename
	print("plotting",resultsFilename)

	results = pd.read_csv("results/901101-910131-SOD-s990.csv")
	results['Optimal market size'] = (results['Optimal buyers']+results['Optimal sellers']) / 2
	results['Normalized market size'] = results['Optimal units'] / (results['Max units per trader'])
	results['log10(M)'] = np.log(results['Max units per trader'])/np.log(10)
	results['MG Satisfaction Ratio'] = (results['Optimal buyers']+results['Optimal sellers']) 
	# results['Optimal buyer ratio'] = (results['Optimal buyers']) / (results['Total buyers']) 
	# results['Optimal seller ratio'] = (results['Optimal sellers']) / (results['Total sellers']) 
	print(len(results), " auctions")
	results = results[results['Optimal gain']>0]
	print(len(results), " auctions with positive optimal gain")
	
	
	# for field in ['MUDA-lottery', 'MUDA-Vickrey traders', 'MUDA-Vickrey total']:
	# 	results[field+' ratio'] = results[field+' gain'] / results['Optimal gain']

	if numOfBins:
		results_bins = results.groupby(pd.cut(results[xColumn],20)).mean()
		print(len(results_bins),results_bins)
		
		print('$$$$')
	else:
		results_bins = results.groupby(results[xColumn]).mean()
		print('****')

	# results_bins.plot(x=xColumn, y='Optimal buyers', style=['b^-'], ax=ax, markersize=markerSize)
	# results_bins.plot(x=xColumn, y='Optimal sellers', style=['ro-'], ax=ax, markersize=markerSize)
	results_bins.plot(x=xColumn, y='MG Satisfaction Ratio', style=['gv-'], ax=ax, markersize=markerSize)

	plt.legend(loc=0,prop={'size':legendFontSize})
	# ax.legend_.remove()
	ax.set_title(title, fontsize= titleFontSize, weight='bold')
	ax.set_ylabel("MG Satisfaction Rate", fontsize= axesFontSize)
	ax.tick_params(axis='both', which='major', labelsize=axesFontSize)
	ax.tick_params(axis='both', which='minor', labelsize=axesFontSize)
	ax.set_ylim(YLIM)


# def plotSocialEfficiencyFile(filename, resultsFilename=None, combineByOrderDate=False, replicaNums=None, agentNums=None, numOfBins=10, ax=None, title=None, xColumn = 'Optimal units'):
# 	if resultsFilename:
# 		pass
# 	elif replicaNums:
# 		resultsFilename = "results/"+\
# 			filename+\
# 			("-combined" if combineByOrderDate else "")+\
# 			"-x"+str(max(replicaNums))+\
# 			".csv"
# 	elif agentNums:
# 		print("THERE")
# 		resultsFilename = "results/"+\
# 			filename+\
# 			("-combined" if combineByOrderDate else "")+\
# 			"-s"+str(max(agentNums))+\
# 			".csv"
# 	else:
# 		print("HERE")
# 		raise(Error("cannot calculate resultsFilename"))
# 	print(resultsFilename)

# 	plotSocialEfficiency(resultsFilename, xColumn=xColumn, numOfBins=numOfBins, ax=ax, title=title)

# def plotSocialEfficiency(resultsFilename, xColumn=xColumn, numOfBins=numOfBins, ax=ax, title=title):
# 	if not ax:
# 		ax = plt.subplot(1, 1, 1)
# 	if not title:
# 		title = resultsFilename
# 	print("plotting",resultsFilename)

# 	results = pd.read_csv("results/901101-910131-SOD-s990.csv")
# 	results['Optimal market size'] = (results['Optimal buyers']+results['Optimal sellers']) / 2
# 	results['Normalized market size'] = results['Optimal units'] / (results['Max units per trader'])
# 	results['log10(M)'] = np.log(results['Max units per trader'])/np.log(10)
# 	results['MG Satisfaction Ratio'] = (results['Optimal buyers']+results['Optimal sellers']) 
# 	# results['Optimal buyer ratio'] = (results['Optimal buyers']) / (results['Total buyers']) 
# 	# results['Optimal seller ratio'] = (results['Optimal sellers']) / (results['Total sellers']) 
# 	print(len(results), " auctions")
# 	results = results[results['Optimal gain']>0]
# 	print(len(results), " auctions with positive optimal gain")
	
	
# 	# for field in ['MUDA-lottery', 'MUDA-Vickrey traders', 'MUDA-Vickrey total']:
# 	# 	results[field+' ratio'] = results[field+' gain'] / results['Optimal gain']

# 	if numOfBins:
# 		results_bins = results.groupby(pd.cut(results[xColumn],20)).mean()
# 		print(len(results_bins),results_bins)
		
# 		print('$$$$')
# 	else:
# 		results_bins = results.groupby(results[xColumn]).mean()
# 		print('****')

# 	results_bins.bar("Total traders", results_bins["MUDA-Vickrey total ratio"], color = "r")
# 	results_bins.bar("Total traders", results["MUDA-Vickrey agent ratio"], color = "b")
# 	results_bins.bar("Total traders", results["MUDA-lottery ratio"], color = "g")
# 	# results_bins.plot(x=xColumn, y='Optimal sellers', style=['ro-'], ax=ax, markersize=markerSize)
# 	# results_bins.plot(x=xColumn, y='MG Satisfaction Ratio', style=['gv-'], ax=ax, markersize=markerSize)

# 	plt.legend(loc=0,prop={'size':legendFontSize})
# 	# ax.legend_.remove()
# 	ax.set_title(title, fontsize= titleFontSize, weight='bold')
# 	ax.set_ylabel("Social Efficiency", fontsize= axesFontSize)
# 	ax.tick_params(axis='both', which='major', labelsize=axesFontSize)
# 	ax.tick_params(axis='both', which='minor', labelsize=axesFontSize)
# 	ax.set_ylim(YLIM)

### MAIN PROGRAM ###

MUDA.LOG = randomTradeWithExogeneousPrice.LOG = False

def torqSimulation():
	numOfBins = 10
	numOfTraderss=list(range(10,1000,10))*1
	filename = "901101-910131-SOD" #"910121-910121-IBM-SOD" #  "901101-910131-SOD" #   "901101-910131- SOD-NORM" # 
	if True:
		# torqSimulateBySymbol(filename, combineByOrderDate=True, agentNums=numOfTraderss)
		torqSimulateBySymbol(filename, combineByOrderDate=False, agentNums=numOfTraderss)
		#torqSimulateBySymbol(filename+"-NORM", combineByOrderDate=False, agentNums=numOfTraderss)
		#torqSimulateBySymbol(filename+"-NORM", combineByOrderDate=True, agentNums=numOfTraderss)
	# plotTorq(filename=filename, combineByOrderDate=False, agentNums=numOfTraderss, numOfBins=numOfBins)
	# plt.show()
	# plotTorq(filename=filename, combineByOrderDate=True, agentNums=numOfTraderss, numOfBins=numOfBins,
	# 		ax = plt.subplot(1,1,1), title="Auctions based on TORQ database", xColumn="Optimal units")
	# plt.xlabel('Optimal #units (k)')
	# plt.show()
	ax = plt.subplot(1,2,1)
	plotTorq(filename=filename, combineByOrderDate=False, agentNums=numOfTraderss, numOfBins=numOfBins, 
			ax=ax, title="Vickrey Double Auction simulation", xColumn="Total traders")
	ax.set_xlabel('Number of traders', fontsize=axesFontSize)
	ax.set_xlim([0,1000])
	ax.set_ylabel(YLABEL, fontsize= axesFontSize)
	ax.tick_params(axis='both', which='major', labelsize=axesFontSize)
	ax.tick_params(axis='both', which='minor', labelsize=axesFontSize)
	ax.set_ylim(YLIM)
	ax = plt.subplot(1,2,2)
	plotLotteryTorq(filename=filename, combineByOrderDate=False, agentNums=numOfTraderss, numOfBins=numOfBins, 
			ax=ax, title="Lottery Double Auction simulation", xColumn="Total traders")
	ax.set_xlabel('Number of traders', fontsize=axesFontSize)
	ax.set_xlim([0,1000])
	ax.set_ylabel(YLABEL, fontsize= axesFontSize)
	ax.tick_params(axis='both', which='major', labelsize=axesFontSize)
	ax.tick_params(axis='both', which='minor', labelsize=axesFontSize)
	ax.set_ylim(YLIM)
	plt.show()

	# ax = plt.subplot(1,1,1)
	# plotSatisfactionFile(filename=filename, combineByOrderDate=False, agentNums=numOfTraderss, numOfBins=numOfBins, 
	# 		ax=ax, title="Microgrid Satisfaction", xColumn="Total traders")
	# ax.set_xlabel('Number of traders', fontsize=axesFontSize)
	# ax.set_xlim([0,1000])
	# ax.set_ylabel("Microgrid Satisfaction Rate", fontsize= axesFontSize)
	# ax.tick_params(axis='both', which='major', labelsize=axesFontSize)
	# ax.tick_params(axis='both', which='minor', labelsize=axesFontSize)
	# ax.set_ylim([0,350])
	

	# plotSocialEfficiencyFile(filename=filename, combineByOrderDate=False, agentNums=numOfTraderss, numOfBins=numOfBins, 
	# 		ax=ax, title="Social Efficiency", xColumn="Total traders")
	# ax.set_xlabel('Number of traders', fontsize=axesFontSize)
	# ax.set_xlim([0,1000])
	# ax.set_ylabel("Social Efficiency", fontsize= axesFontSize)
	# ax.tick_params(axis='both', which='major', labelsize=axesFontSize)
	# ax.tick_params(axis='both', which='minor', labelsize=axesFontSize)
	# ax.set_ylim([0,1])
	# ax = plt.subplot(1,2,1, sharey=None)
	# plotTorq(filename=filename, combineByOrderDate=False, agentNums=numOfTraderss, numOfBins=numOfBins, ax=ax, title="TORQ; additive", xColumn="Total traders")
	# ax.set_xlabel('Total #traders', fontsize=axesFontSize)
	# ax.set_xlim([0,1000])
	# plt.show()

	# ax = plt.subplot(1,2,2)
	# plotTorq(filename=filename+"-NORM", combineByOrderDate=True, agentNums=numOfTraderss, numOfBins=numOfBins, 
	# 		ax=ax, title="TORQ; normalized, combined", xColumn="Total traders")
	# ax.set_xlabel('Total #traders', fontsize=axesFontSize)
	# ax.set_xlim([0,1000])
	# ax = plt.subplot(1,2,1, sharey=ax)
	# plotTorq(filename=filename+"-NORM", combineByOrderDate=False, agentNums=numOfTraderss, numOfBins=numOfBins, ax=ax, title="TORQ; normalized, additive", xColumn="Total traders")
	# ax.set_xlabel('Total #traders', fontsize=axesFontSize)
	# ax.set_xlim([0,1000])
	# plt.show()
	# plotTorq(filename=filename+"-NORM", combineByOrderDate=True, agentNums=numOfTraderss, numOfBins=numOfBins,
	# 		ax = plt.subplot(2,1,2))
	# plt.show()

def randomSimulation(numOfAuctions = 10):
	numOfTraderss = range(100, 1100, 100)
	minNumOfUnitsPerTrader = 10
	maxNumOfUnitsPerTraders = [100,200,500,1000,10,20,50]
	meanValue = 500
	maxNoiseSizes = [50,100,150,200,300,350,400,450,500,250]
	numOfBins = 0
	
	# general
	filenameTraders = "results/random-traders-{}units-{}noise.csv".format(maxNumOfUnitsPerTraders[-1],maxNoiseSizes[-1])
	filenameUnitsFixedTraders   = "results/random-units-{}traders-{}noise.csv".format(numOfTraderss[-1],maxNoiseSizes[-1])
	filenameUnitsFixedVirtual   = "results/random-units-{}virtual-{}noise.csv".format(numOfTraderss[-1],maxNoiseSizes[-1])
	filenameNoise   = "results/random-noise-{}traders-{}units.csv".format(numOfTraderss[-1],maxNumOfUnitsPerTraders[1])
	
	# additive
	# filenameTradersAdd = "results/random-traders-{}units-{}noise-additive.csv".format(maxNumOfUnitsPerTraders[3],maxNoiseSizes[-1])
	# filenameUnitsAdd   = "results/random-units-{}traders-{}noise-additive.csv".format(numOfTraderss[-1],maxNoiseSizes[-1])
	# filenameNoiseAdd   = "results/random-noise-{}traders-{}units-additive.csv".format(numOfTraderss[-1],maxNumOfUnitsPerTraders[1])
	if True:
		keyColumns=("numOfTraders","minNumOfUnitsPerTrader","maxNumOfUnitsPerTrader","maxNoiseSize")

		### non-additive
		simulateAuctions(randomAuctions(     ### as function of #traders
			numOfAuctions, numOfTraderss, minNumOfUnitsPerTrader, maxNumOfUnitsPerTraders[-1:], meanValue, maxNoiseSizes[-1:], fixedNumOfVirtualTraders=True),
			filenameTraders, keyColumns=keyColumns)
		simulateAuctions(randomAuctions(     ### as function of m - fixed total units
			numOfAuctions, numOfTraderss[-1:], minNumOfUnitsPerTrader, maxNumOfUnitsPerTraders, meanValue, maxNoiseSizes[-1:], fixedNumOfVirtualTraders=True),
			filenameUnitsFixedVirtual, keyColumns=keyColumns)
		# simulateAuctions(randomAuctions(   ### as function of m - fixed total traders - TOO LONG
		# 	numOfAuctions, [100], minNumOfUnitsPerTrader, maxNumOfUnitsPerTraders, meanValue, maxNoiseSizes[-1:], fixedNumOfVirtualTraders=False),
		# 	filenameUnitsFixedTraders, keyColumns=keyColumns)
		simulateAuctions(randomAuctions(     ### as function of noise
			numOfAuctions, numOfTraderss[-1:], minNumOfUnitsPerTrader, maxNumOfUnitsPerTraders[-1:], meanValue, maxNoiseSizes, fixedNumOfVirtualTraders=True),
			filenameNoise, keyColumns=keyColumns)

		### additive
		# simulateAuctions(randomAuctions(  ### as function of #traders
		# 	numOfAuctions, numOfTraderss, maxNumOfUnitsPerTraders[3], maxNumOfUnitsPerTraders[-1:], meanValue, maxNoiseSizes[-1:], fixedNumOfVirtualTraders=True),
		# 	filenameTradersAdd, keyColumns=keyColumns)
		# # simulateAuctions(randomAuctions(   ### as function of m - fixed total units
		# # 	numOfAuctions, numOfTraderss[-1:], maxNumOfUnitsPerTraders, meanValue, maxNoiseSizes[-1:],isAdditive=True, fixedNumOfVirtualTraders=True),
		# # 	filenameUnitsAdd, keyColumns=keyColumns)
		# simulateAuctions(randomAuctions(    ### as function of noise
		# 	numOfAuctions, numOfTraderss[-1:],  maxNumOfUnitsPerTraders[3], maxNumOfUnitsPerTraders[-1:], meanValue, maxNoiseSizes, fixedNumOfVirtualTraders=True),
		# 	filenameNoiseAdd, keyColumns=keyColumns)
		# # simulateAuctions(randomAuctions(   ### as function of m - fixed total traders
		# # 	numOfAuctions, [100], maxNumOfUnitsPerTraders, meanValue, maxNoiseSizes[-1:], isAdditive=True, fixedNumOfVirtualTraders=False),
		# # 	filenameUnitsFixedTraders, keyColumns=keyColumns)

	TITLESTART = ""# "Uniform; "
	### non-additive
	ax=plt.subplot(1,2,1)
	plotResults(filenameTraders,"Total traders",numOfBins, ax, title=
		TITLESTART+"m={},M={},noise={}".format(minNumOfUnitsPerTrader,maxNumOfUnitsPerTraders[-1],maxNoiseSizes[-1]))
	ax.set_xlabel('Total #traders', fontsize=axesFontSize)
	ax.set_xlim([0,1000])


	# ax=plt.subplot(1,1,1)
	# plotResults(filenameTraders,"Optimal units",numOfBins, ax, title=
	#	TITLESTART+"m={},M={},noise={}".format(minNumOfUnitsPerTrader,maxNumOfUnitsPerTraders[3],maxNoiseSizes[-1]))
	# plt.xlabel('Optimal #units (k)')
	# plt.show()

	ax=plt.subplot(1,2,2, sharey=None)
	plotResults(filenameUnitsFixedVirtual,"log10(M)",numOfBins=None, ax=ax, title=TITLESTART+"m={},units={},noise={}".format(minNumOfUnitsPerTrader,numOfTraderss[-1],maxNoiseSizes[-1]))
	#labels = [""]+["{:.0e}".format(t) for t in sorted(maxNumOfUnitsPerTraders)]
	ax.set_xlim([1,8])
	ax.set_xticklabels(["","100","1e3","1e4","1e5","1e6","1e7","1e8"])
	ax.set_xlabel('Max #units per trader (M)', fontsize=axesFontSize)
	ax.set_ylabel("")
	
	plt.show()
	# plotResults(filenameUnitsFixedTraders,"maxNumOfUnitsPerTrader",numOfBins, plt.subplot(1,1,1), 
	# 	title="traders={}, noise={}".format(numOfTraderss[-1],maxNoiseSizes[-1]))
	# plt.xlabel('#units per trader (M)')
	# plt.show()
	# plotResults(filenameNoise,"maxNoiseSize",numOfBins, plt.subplot(1,1,1), 
	# 	title=TITLESTART+"units={},m={},M={}".format(numOfTraderss[-1],minNumOfUnitsPerTrader,maxNumOfUnitsPerTraders[3]))
	# plt.xlabel('Max noise size (A)', fontsize=axesFontSize)
	# plt.show()

	### additive
	# # plotResults(filenameTradersAdd,"numOfTraders",numOfBins, plt.subplot(1,1,1), title=TITLESTART+"m={},M={},n oise={}, additive".format(minNumOfUnitsPerTrader,maxNumOfUnitsPerTraders[3],maxNoiseSizes[-1]))
	# # plt.xlabel('total #units')
	# plotResults(filenameTradersAdd,"Optimal units",numOfBins, plt.subplot(1,1,1),
	# 	title=TITLESTART+"m={},M={},noise={},additive".format(minNumOfUnitsPerTrader, maxNumOfUnitsPerTraders[3],maxNoiseSizes[-1]))
	# plt.xlabel('optimal #units (k)')
	# plt.show()
	# 
	# # plotResults(filenameUnitsAdd,"maxNumOfUnitsPerTrader",numOfBins, plt.subplot(1,1,1), 
	# # 	title=TITLESTART+"traders={},noise={},additive".format(numOfTraderss[-1],maxNoiseSizes[-1]))
	# # plt.ylabel('')
	# plotResults(filenameNoiseAdd,"maxNoiseSize",numOfBins, plt.subplot(1,1,1), 
	# 	title=TITLESTART+"traders={},m={},M={},additive".format(numOfTraderss[-1],minNumOfUnitsPerTrader, maxNumOfUnitsPerTraders[3]))
	# plt.xlabel('Max noise size (A)')
	# plt.show()


createResults = False # True # 

torqSimulation()
# randomSimulation(numOfAuctions = 10)
