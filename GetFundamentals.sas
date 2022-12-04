%let startdt = '01Dec1999'd;
%let enddt = '31Oct2020'd;

data work.tickers;
    input ticker $;
    datalines;
AAPL
ABC
ABMD
ABT
ADBE
ADI
ADM
ADP
ADSK
AEE
AEP
AES
AFL
AIG
AIV
AJG
AKAM
ALB
ALK
ALL
AMAT
AMD
AME
AMGN
AMT
AMZN
ANSS
AON
AOS
APA
APD
APH
ARE
ATO
ATVI
AVB
AVY
AXP
AZO
BA
BAC
BAX
BBY
BDX
BEN
BF-B
BK
BKNG
BKR
BLK
BLL
BMY
BRK-B
BSX
BWA
BXP
C
CAG
CAH
CAT
CB
CCI
CCL
CDNS
CERN
CHD
CHRW
CI
CINF
CL
CLX
CMA
CMCSA
CMI
CMS
CNP
COF
COG
COO
COP
COST
CPB
CPRT
CSCO
CSX
CTAS
CTSH
CTXS
CVS
CVX
D
DD
DE
DGX
DHI
DHR
DIS
DISH
DLTR
DOV
DRE
DRI
DTE
DUK
DVA
DVN
DXC
EA
EBAY
ECL
ED
EFX
EIX
EL
EMN
EMR
EOG
EQR
ES
ESS
ETN
ETR
EVRG
EXC
EXPD
F
FAST
FCX
FDX
FE
FFIV
FISV
FITB
FLS
FMC
FRT
GD
GE
GIS
GL
GLW
GPC
GPS
GS
GWW
HAL
HAS
HBAN
HD
HES
HFC
HIG
HOG
HOLX
HON
HP
HPQ
HRB
HRL
HSIC
HST
HSY
HUM
IBM
IDXX
IEX
IFF
INCY
INTC
INTU
IP
IPG
IRM
IT
ITW
IVZ
J
JBHT
JCI
JKHY
JNJ
JNPR
JPM
JWN
K
KEY
KIM
KLAC
KMB
KMX
KO
KR
KSS
KSU
L
LEG
LEN
LH
LHX
LIN
LLY
LMT
LNC
LNT
LOW
LRCX
LUV
M
MAA
MAR
MAS
MCD
MCHP
MCK
MCO
MDT
MGM
MHK
MKC
MLM
MMC
MMM
MO
MOS
MRK
MRO
MS
MSFT
MSI
MTB
MTD
MU
MXIM
NEE
NEM
NI
NKE
NLOK
NOC
NOV
NSC
NTAP
NTRS
NUE
NVDA
NVR
NWL
O
OKE
OMC
ORCL
ORLY
OXY
PAYX
PBCT
PCAR
PEAK
PEG
PEP
PFE
PG
PGR
PH
PHM
PKI
PLD
PNC
PNR
PNW
PPG
PPL
PRGO
PSA
PVH
PWR
PXD
QCOM
RCL
RE
REG
REGN
RF
RHI
RJF
RL
RMD
ROK
ROL
ROP
ROST
RSG
RTX
SBAC
SBUX
SCHW
SEE
SHW
SIVB
SJM
SLB
SLG
SNA
SNPS
SO
SPG
SPGI
SRE
STE
STT
STZ
SWK
SWKS
SYK
SYY
T
TAP
TFC
TFX
TGT
TJX
TMO
TROW
TRV
TSCO
TSN
TT
TXN
TXT
UDR
UHS
UNH
UNM
UNP
URI
USB
VFC
VLO
VMC
VNO
VRSN
VRTX
VZ
WAB
WAT
WBA
WDC
WEC
WELL
WFC
WHR
WM
WMB
WMT
WRB
WY
XEL
XLNX
XOM
XRAY
XRX
YUM
ZBRA
ZION
;
run;

proc sql;
create table out as 
select datadate, tic, SRCQ,
ACTQ,  /*ACTQ -- Current Assets - Total (ACTQ)*/ 
ANCQ,  /*ANCQ -- Non-Current Assets - Total (ANCQ)*/ 
ATQ,  /*ATQ -- Assets - Total (ATQ)*/ 
CEQQ,  /*CEQQ -- Common/Ordinary Equity - Total (CEQQ)*/ 
CHEQ,  /*CHEQ -- Cash and Short-Term Investments (CHEQ)*/ 
CHQ,  /*CHQ -- Cash (CHQ)*/ 
COGSQ,  /*COGSQ -- Cost of Goods Sold (COGSQ)*/ 
CSHIQ,  /*CSHIQ -- Common Shares Issued (CSHIQ)*/ 
CSHOPQ,  /*CSHOPQ -- Total Shares Repurchased - Quarter (CSHOPQ)*/ 
CSHOQ,  /*CSHOQ -- Common Shares Outstanding (CSHOQ)*/ 
CSTKQ,  /*CSTKQ -- Common/Ordinary Stock (Capital) (CSTKQ)*/ 
DLCQ,  /*DLCQ -- Debt in Current Liabilities (DLCQ)*/ 
DLTTQ,  /*DLTTQ -- Long-Term Debt - Total (DLTTQ)*/ 
DPQ,  /*DPQ -- Depreciation and Amortization - Total (DPQ)*/ 
EPSPI12,  /*EPSPI12 -- Earnings Per Share (Basic) - Including Extraordinary Items - 12 Months Movi (EPSPI12)*/ 
EPSPIQ,  /*EPSPIQ -- Earnings Per Share (Basic) - Including Extraordinary Items (EPSPIQ)*/ 
EPSPXQ,  /*EPSPXQ -- Earnings Per Share (Basic) - Excluding Extraordinary Items (EPSPXQ)*/ 
INTANQ,  /*INTANQ -- Intangible Assets - Total (INTANQ)*/ 
INVTQ,  /*INVTQ -- Inventories - Total (INVTQ)*/ 
LCTQ,  /*LCTQ -- Current Liabilities - Total (LCTQ)*/ 
LLTQ,  /*LLTQ -- Long-Term Liabilities (Total) (LLTQ)*/ 
LTQ,	/* LTQ -- Liabilities - Total (LTQ)*/
NIQ,  /*NIQ -- Net Income (Loss) (NIQ)*/ 
OIBDPQ,  /*OIBDPQ -- Operating Income Before Depreciation - Quarterly (OIBDPQ)*/ 
OPEPSQ,  /*OPEPSQ -- Earnings Per Share from Operations (OPEPSQ)*/ 
PLLQ,  /*PLLQ -- Provision for Loan/Asset Losses (PLLQ)*/ 
PPEGTQ,  /*PPEGTQ -- Property, Plant and Equipment - Total (Gross) - Quarterly (PPEGTQ)*/ 
RECDQ,  /*RECDQ -- Receivables - Estimated Doubtful (RECDQ)*/ 
RECTQ,  /*RECTQ -- Receivables - Total (RECTQ)*/ 
REQ,  /*REQ -- Retained Earnings (REQ)*/ 
SALEQ,  /*SALEQ -- Sales/Turnover (Net) (SALEQ)*/ 
TEQQ,  /*TEQQ -- Stockholders Equity - Total (TEQQ)*/ 
CAPXY,  /*CAPXY -- Capital Expenditures (CAPXY)*/ 
DEPCY,  /*DEPCY -- Depreciation and Depletion (Cash Flow) (DEPCY)*/ 
DLCCHY,  /*DLCCHY -- Changes in Current Debt (DLCCHY)*/ 
DLTISY,  /*DLTISY -- Long-Term Debt - Issuance (DLTISY)*/ 
DLTRY,  /*DLTRY -- Long-Term Debt - Reduction (DLTRY)*/ 
NIY,  /*NIY -- Net Income (Loss) (NIY)*/ 
OANCFY,  /*OANCFY -- Operating Activities - Net Cash Flow (OANCFY)*/ 
TXPDY,  /*TXPDY -- Income Taxes Paid (TXPDY)*/ 
TXTY,  /*TXTY -- Income Taxes - Total (TXTY)*/ 
DVPSPQ,  /*DVPSPQ -- Dividends per Share - Pay Date - Quarter (DVPSPQ)*/ 
MKVALTQ,  /*MKVALTQ -- Market Value - Total (MKVALTQ)*/ 
PRCCQ  /*PRCCQ -- Price Close - Quarter (PRCCQ)*/ 
from compd.fundq
inner join 
tickers on tic = ticker
where consol = 'C' and popsrc = 'D' and datadate between &startdt. and &enddt.
and SRCQ not in (88);
quit;
/*  */
/* proc sort data=out; */
/* by datadate tic; */
/* run; */
/*  */
/* proc sql; */
/* create table test as */
/* select a.* from out a inner join  */
/* out b on a.tic = b.tic and a.datadate = b.datadate */
/* where a.SRCQ in (88) and b.SRCQ not in (88); */
/* quit; */

proc sql;
select a.* from compd.fundq a inner join 
(select datadate, tic from out group by datadate, tic having(count(*) > 1)) b on 
a.datadate = b.datadate and a.tic = b.tic;
quit;
