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
	'Optimal gain', 'MIDA-lottery gain', 'MIDA-Vickrey traders gain', 'MUDA-Vickrey total gain')

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
	columns = keyColumns+COLUMNS
	results = DataFrame(columns=columns)
	print("\t{}".format(columns))
	resultsFilenameTemp = resultsFilename+".temp"
	for auctionID,traders in auctions:
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
		(sizeMUDALottery, gainMUDALottery, gainMUDALottery, sizeMUDAVickrey, tradersGainMUDAVickrey, totalGainMUDAVickrey) = MUDA(traders, Lottery=True, Vickrey=True)
		resultsRow = [
			*auctionID,
			totalBuyers, totalSellers, totalBuyers+totalSellers, min(totalBuyers,totalSellers), sum(unitsPerTrader),
			maxUnitsPerTrader, minUnitsPerTrader, maxUnitsPerTrader/max(1,minUnitsPerTrader), stddev,
			buyersWALRAS, sellersWALRAS, sizeWALRAS,
			gainWALRAS, gainMUDALottery, tradersGainMUDAVickrey, totalGainMUDAVickrey]
		print("\t{}".format(resultsRow))
		results.loc[len(results)] = resultsRow
		results.to_csv(resultsFilenameTemp)
	results.to_csv(resultsFilename)
	os.remove(resultsFilenameTemp)
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

YLABEL = 'MUDA GFT divided by maximum GFT'
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

	plotResults(resultsFilename, xColumn=xColumn, numOfBins=numOfBins, ax=ax, title=title)


def plotResults(resultsFilename=None, xColumn='Min total traders', numOfBins=10, ax=None, title=None):
	if not ax:
		ax = plt.subplot(1, 1, 1)
	if not title:
		title = resultsFilename
	print("plotting",resultsFilename)

	results = pd.read_csv(resultsFilename)
	results['Optimal market size'] = (results['Optimal buyers']+results['Optimal sellers']) / 2
	results['Normalized market size'] = results['Optimal units'] / (results['Max units per trader'])
	results['log10(M)'] = np.log(results['Max units per trader'])/np.log(10)
	print(len(results), " auctions")
	results = results[results['Optimal gain']>0]
	print(len(results), " auctions with positive optimal gain")

	for field in ['MIDA-lottery', 'MIDA-Vickrey traders', 'MIDA-Vickrey total']:
		fieldNew = field.replace("MIDA","MUDA")
		results[fieldNew+' ratio'] = results[field+' gain'] / results['Optimal gain']

	if numOfBins:
		results_bins = results.groupby(pd.cut(results[xColumn],numOfBins)).mean()
	else:
		results_bins = results.groupby(results[xColumn]).mean()

	results_bins.to_csv(resultsFilename+".bins")

	results_bins.plot(x=xColumn, y='MUDA-Vickrey total ratio', style=['b^-'], ax=ax, markersize=markerSize)
	results_bins.plot(x=xColumn, y='MUDA-Vickrey traders ratio', style=['gv-'], ax=ax, markersize=markerSize)
	results_bins.plot(x=xColumn, y='MUDA-lottery ratio', style=['ro-'], ax=ax, markersize=markerSize)

	plt.legend(loc=0,prop={'size':legendFontSize})
	#ax.legend_.remove()
	ax.set_title(title, fontsize= titleFontSize, weight='bold')
	ax.set_ylabel(YLABEL, fontsize= axesFontSize)
	ax.tick_params(axis='both', which='major', labelsize=axesFontSize)
	ax.tick_params(axis='both', which='minor', labelsize=axesFontSize)
	ax.set_ylim(YLIM)


### MAIN PROGRAM ###

MUDA.LOG = randomTradeWithExogeneousPrice.LOG = False

def torqSimulation():
	numOfBins = 100
	numOfTraderss=list(range(10,1000,10))*1
	filename = "901101-910131-SOD"
	if createResults:
		torqSimulateBySymbol(filename, combineByOrderDate=True, agentNums=numOfTraderss)
		torqSimulateBySymbol(filename, combineByOrderDate=False, agentNums=numOfTraderss)
	ax = plt.subplot(1,2,2)
	plotTorq(filename=filename, combineByOrderDate=True, agentNums=numOfTraderss, numOfBins=numOfBins,
			ax=ax, title="TORQ; combined", xColumn="Total traders")
	ax.set_xlabel('Total #traders', fontsize=axesFontSize)
	ax.set_xlim([0,1000])
	ax.set_ylabel("")

	ax = plt.subplot(1,2,1, sharey=None)
	plotTorq(filename=filename, combineByOrderDate=False, agentNums=numOfTraderss, numOfBins=numOfBins, ax=ax, title="TORQ; additive", xColumn="Total traders")
	ax.set_xlabel('Total #traders', fontsize=axesFontSize)
	ax.set_xlim([0,1000])
	plt.show()


def randomSimulation(numOfAuctions = 100):
	numOfTraderss = range(2000000, 42000000, 2000000) #range(200,4200,200) #
	minNumOfUnitsPerTrader = 1 # 10
	maxNumOfUnitsPerTraders = [100,1000,10000,100000,1000000,10000000,100000000,10]
	meanValue = 500
	maxNoiseSizes = [50,100,150,200,300,350,400,450,500,250]
	numOfBins = 20

	# general
	filenameTraders = "results/random-traders-{}units-{}noise.csv".format(maxNumOfUnitsPerTraders[-1],maxNoiseSizes[-1])
	filenameUnitsFixedTraders   = "results/random-units-{}traders-{}noise.csv".format(numOfTraderss[-1],maxNoiseSizes[-1])
	filenameUnitsFixedVirtual   = "results/random-units-{}virtual-{}noise.csv".format(numOfTraderss[-1],maxNoiseSizes[-1])
	filenameNoise   = "results/random-noise-{}traders-{}units.csv".format(numOfTraderss[-1],maxNumOfUnitsPerTraders[3])

	# additive
	filenameTradersAdd = "results/random-traders-{}units-{}noise-additive.csv".format(maxNumOfUnitsPerTraders[3],maxNoiseSizes[-1])
	filenameUnitsAdd   = "results/random-units-{}traders-{}noise-additive.csv".format(numOfTraderss[-1],maxNoiseSizes[-1])
	filenameNoiseAdd   = "results/random-noise-{}traders-{}units-additive.csv".format(numOfTraderss[-1],maxNumOfUnitsPerTraders[3])
	if createResults:
		keyColumns=("numOfTraders","minNumOfUnitsPerTrader","maxNumOfUnitsPerTrader","maxNoiseSize")

		# non-additive
		simulateAuctions(randomAuctions(     ### as function of #traders
			numOfAuctions, numOfTraderss, minNumOfUnitsPerTrader, maxNumOfUnitsPerTraders[-1:], meanValue, maxNoiseSizes[-1:], fixedNumOfVirtualTraders=True),
			filenameTraders, keyColumns=keyColumns)
		simulateAuctions(randomAuctions(     ### as function of m - fixed total units
			numOfAuctions, numOfTraderss[-1:], minNumOfUnitsPerTrader, maxNumOfUnitsPerTraders, meanValue, maxNoiseSizes[-1:], fixedNumOfVirtualTraders=True),
			filenameUnitsFixedVirtual, keyColumns=keyColumns)

	TITLESTART = ""
	### non-additive
	ax=plt.subplot(1,2,1)
	plotResults(filenameTraders,"Total traders",numOfBins, ax, title=
		TITLESTART+"Fixed units per trader, M={}".format(maxNumOfUnitsPerTraders[-1]))
	ax.set_xlabel('Total #traders', fontsize=axesFontSize)

	ax=plt.subplot(1,2,2, sharey=None)
	plotResults(filenameUnitsFixedVirtual,"log10(M)",numOfBins=None, ax=ax, title=TITLESTART+"Fixed total units, M*n={}".format(numOfTraderss[-1]))
	ax.set_xlim([1,8])
	ax.set_xticklabels(["","100","1e3","1e4","1e5","1e6","1e7","1e8"])
	ax.set_xlabel('Max #units per trader (M)', fontsize=axesFontSize)
	ax.set_ylabel("")

	plt.show()


createResults = False # True # 

torqSimulation()
#randomSimulation(numOfAuctions = 10)
