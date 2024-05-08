'''
1) endowments and Endowment assurance is done. For Joint Life the code is commented out due to errors. However, the
Logic in the code is applied.

2)Lapse is done for whole-life and term-life assurance using the Poisson distribution, where the probability of lapsing
is either 0 or 1 at the date premium. PV_in, PV_out, and cashflows were affected because of lapses where
All the necessary adjustments are made. Adjustments are also made for unusual data errors that might come.

3)PV of Liabitites is taken out by taking project date as 2024.

4) A new function is created where you need to give an asset value at the beginning, and then it splits into the correct
propionate along with rebalancing at the start of the year or whenever the cash balance is less than 0.

5) The percentile method is used with the correct average value in order to predict.

6 ) interest rates and inflation are rates are updates with Ornstien Uhbleck continous model

7 ) Typos in the code that was in the videos related to gender was corrected

8) A new function is added because the dates in the file used - instead of /
'
'''

import datetime
import random
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt



class TbaseTable:
    def __init__(self, xlFile):
        temp = pd.read_excel(xlFile).to_numpy()
        self.lxm = np.append(temp[:, 1], [0, 0])
        self.lxf = np.append(temp[:, 2], [0, 0])
        self.age = temp[:, 0]

    def lx(self, x, gender):  # gender is 0 for male  , 1 for female
        if x >= 101:
            return 0
        if x < 0:
            return 100000
        x1 = int(x)
        x2 = x1 + 1
        if gender == 0:
            l1 = self.lxm[x1]
            l2 = self.lxm[x2]
        elif gender == 1:
            l1 = self.lxf[x1]
            l2 = self.lxf[x2]
        else:
            return None
        frac = x - x1
        lx_value = (1 - frac) * l1 + frac * l2
        return lx_value

    def tpx(self, t, x, gender):
        if x > 101:
            return 0
        return self.lx(x + t, gender) / self.lx(x, gender)


class TlifeTable:
    def __init__(self, baseTable, interest):
        self.interest = interest
        self.baseTable = baseTable

    def lx(self, x, gender):
        return self.baseTable.lx(x, gender)

    def tpx(self, t, x, gender):
        return self.baseTable.tpx(t, x, gender)

    def tqx(self, t, x, gender):
        return 1 - self.tpx(t, x, gender)

    def setup_annuities(self):
        self.anndotm = np.empty([102], dtype=float)
        self.anndotm[101] = 0
        self.anndotf = np.empty([102], dtype=float)
        self.anndotf[101] = 0
        for x in range(100, -1, -1):
            self.anndotm[x] = 1 + self.anndotm[x + 1] * self.tpx(1, x, 0) / (1 + self.interest)
            self.anndotf[x] = 1 + self.anndotf[x + 1] * self.tpx(1, x, 1) / (1 + self.interest)

    def anndot(self, age, gender):
        if age >= 101:
            return 1
        iage = int(age)
        fage = age - iage
        if gender == 0:
            ann = self.anndotm[iage] * (1 - fage) + self.anndotm[iage + 1] * fage
        else:
            ann = self.anndotf[iage] * (1 - fage) + self.anndotf[iage + 1] * fage
        return ann

    def setup_assurances(self):
        self.asscapm = np.empty([102], dtype=float)
        self.asscapm[101] = 1 / (1 + self.interest)
        self.asscapf = np.empty([102], dtype=float)
        self.asscapf[101] = 1 / (1 + self.interest)
        for x in range(100, -1, -1):
            self.asscapm[x] = (self.asscapm[x + 1] * self.tpx(1, x, 0) + (self.tqx(1, x, 0))) / (1 + self.interest)
            self.asscapf[x] = (self.asscapf[x + 1] * self.tpx(1, x, 1) + (self.tqx(1, x, 1))) / (1 + self.interest)

    def asscap(self, age, gender):
        iage = int(age)
        if age >= 101:
            return 1  # instant death
        fage = age - iage
        if gender == 0:
            ass = self.asscapm[iage] * (1 - fage) + self.asscapm[iage + 1] * fage
        else:
            ass = self.asscapf[iage] * (1 - fage) + self.asscapf[iage + 1] * fage
        return ass

    def setup_endowment(self):
        self.endom = np.empty([102], dtype=float)
        self.endom[101] = 1 / (1 + self.interest)
        self.endof = np.empty([102], dtype=float)
        self.endof[101] = 1 / (1 + self.interest)
        for x in range(100, -1, -1):
            self.endom[x] = (self.endom[x + 1] * self.tpx(1, x, 0)) / (1 + self.interest)
            self.endof[x] = (self.endof[x + 1] * self.tpx(1, x, 1)) / (1 + self.interest)

    def endow(self, age, gender):
        iage = int(age)
        if age >= 101:
            return 1  # instant death
        fage = age - iage
        if gender == 0:
            endo = self.endom[iage] * (1 - fage) + self.endom[iage + 1] * fage
        else:
            endo = self.endof[iage] * (1 - fage) + self.endof[iage + 1] * fage
        return endo

    def setup_assurances_joint(self):
        self.asscap_joint_first = np.empty((102, 102), dtype=float)
        self.asscap_joint_first.fill(np.nan)
        self.asscap_joint_first[101, :] = 1 / (1 + self.interest)  # Set the last row for first deaths
        self.asscap_joint_first[:, 101] = 1 / (1 + self.interest)  # Set the last column for first deaths

        self.asscap_joint_second = np.empty((102, 102), dtype=float)
        self.asscap_joint_second.fill(np.nan)
        self.asscap_joint_second[101, 101] = 1 / (1 + self.interest)  # Only the last cell for second deaths

        # Iterate over each combination of ages for two individuals from 100 down to 0
        for x in range(100, -1, -1):
            for y in range(100, -1, -1):
                joint_survival = self.tpx(1, x, 0) * self.tpx(1, y, 1)
                if x <= 100 or y <= 100:  # Avoid out of bounds when x or y equals 100.
                    self.asscap_joint_first[x][y] = (self.asscap_joint_first[x + 1][y + 1] * joint_survival + (
                            1 - (self.tpx(1, x, 0) * self.tpx(1, y, 1)))) / (1 + self.interest)
                    self.asscap_joint_second[x][y] = (self.asscap_joint_second[x + 1][y + 1] * joint_survival + (
                            self.tqx(1, x, 0) + self.tqx(1, y, 1) - self.tqx(1, x, 0) * self.tqx(1, y, 1))) / (
                                                             1 + self.interest)

    def asscap_joint_first(self, age_x, age_y):
        iage_x = int(age_x)
        fage_x = age_x - int(age_x)
        iage_y = int(age_y)
        fage_y = age_y - int(age_x)

        ass_joint = (self.asscap_joint_first[iage_x][iage_y] * (1 - fage_x) * (1 - fage_y)
                     + self.asscap_joint_first[iage_x][iage_y + 1] * (1 - fage_x) * fage_y
                     + self.asscap_joint_first[iage_x + 1][iage_y] * fage_x * (1 - fage_y)
                     + self.asscap_joint_first[iage_x + 1][iage_y + 1] * fage_x * fage_y)
        return ass_joint

    def asscap_joint_second(self, age_x, age_y):
        iage_x, fage_x = int(age_x), age_x - int(age_x)
        iage_y, fage_y = int(age_y), age_y - int(age_y)

        ass_joint = (self.asscap_joint_second[iage_x][iage_y] * (1 - fage_x) * (1 - fage_y)
                     + self.asscap_joint_second[iage_x][iage_y + 1] * (1 - fage_x) * fage_y
                     + self.asscap_joint_second[iage_x + 1][iage_y] * fage_x * (1 - fage_y)
                     + self.asscap_joint_second[iage_x + 1][iage_y + 1] * fage_x * fage_y)
        return ass_joint


class TsuperTable:
    def __init__(self, name, xlFile, interest1, interest2, interestStep):
        self.name = name
        self.interest1 = interest1
        self.interest2 = interest2
        self.interestStep = interestStep
        self.baseTable = TbaseTable(xlFile)
        self.allTables = []
        print("Setup actuarial tables")
        for i in np.arange(interest1, interest2, interestStep):
            newTable = TlifeTable(self.baseTable, i)
            newTable.setup_annuities()
            newTable.setup_assurances()
            newTable.setup_endowment()
           # newTable.setup_assurances_joint()
            self.allTables.append(newTable)

    def annuity(self, age, gender, startOffset, endOffset, pre_interest, interest, pthly):
        if interest < self.interest1:
            i_index = 0
        elif interest >= self.interest2 - self.interestStep * 2:
            fake_interest = self.interest1 - self.interestStep * 2
            i_index = int((fake_interest - self.interest1) / self.interestStep)
        else:
            i_index = int((interest - self.interest1) / self.interestStep)
        i1 = i_index * self.interestStep + self.interest1
        i2 = i1 + self.interestStep
        ifrac = (interest - i1) / (i2 - i1)
        table1 = self.allTables[i_index]
        table2 = self.allTables[i_index + 1]
        ageStart = age + startOffset
        lifeAnnuity_1 = table1.anndot(ageStart, gender)
        lifeAnnuity_2 = table2.anndot(ageStart, gender)
        n_payments = (endOffset - startOffset) * pthly
        n_payments = math.ceil(n_payments)
        ageEnd = age + startOffset + n_payments / pthly
        new_endoffset = ageEnd - age
        endAnnuity_1 = table1.anndot(ageEnd, gender)
        endAnnuity_2 = table2.anndot(ageEnd, gender)
        pthly_adjust = (pthly - 1) / (2 * pthly)
        lifeAnnuity = lifeAnnuity_1 * (1 - ifrac) + lifeAnnuity_2 * ifrac - pthly_adjust
        endAnnuity = endAnnuity_1 * (1 - ifrac) + endAnnuity_2 * ifrac - pthly_adjust
        startDiscount = table1.tpx(startOffset, age, gender)
        endDiscount = (1 + interest) ** (startOffset - new_endoffset) * table1.tpx(new_endoffset, age, gender)
        total_value = (1 + pre_interest) ** (-startOffset) * (startDiscount * (lifeAnnuity) - endDiscount * (endAnnuity))
        return total_value

    def assurance(self, age, gender, startOffset, endOffset, interest, deferral):
        if interest < self.interest1:
            i_index = 0
        elif interest >= self.interest2 - self.interestStep * 2:
            fake_interest = self.interest2 - self.interestStep * 2
            i_index = int((fake_interest - self.interest1) / self.interestStep)
        else:
            i_index = int((interest - self.interest1) / self.interestStep)
        i1 = i_index * self.interestStep + self.interest1
        i2 = i1 + self.interestStep
        ifrac = (interest - i1) / (i2 - i1)
        table1 = self.allTables[i_index]
        table2 = self.allTables[i_index + 1]
        ageStart = age + startOffset
        startAssurance_1 = table1.asscap(ageStart, gender)
        startAssurance_2 = table2.asscap(ageStart, gender)
        ageEnd = age + endOffset
        endAssurance_1 = table1.asscap(ageEnd, gender)
        endAssurance_2 = table2.asscap(ageEnd, gender)
        startAssurance = startAssurance_1 * (1 - ifrac) + startAssurance_2 * ifrac
        endAssurance = endAssurance_1 * (1 - ifrac) + endAssurance_2 * ifrac
        startDiscount = (1 + interest) ** (-startOffset) * table1.tpx(startOffset, age, gender)
        endDiscount = (1 + interest) ** (-endOffset) * table1.tpx(endOffset, age, gender)
        total_value = startDiscount * startAssurance - endDiscount * endAssurance
        total_value = total_value * (1 + interest) ** (0.5 - deferral)
        return total_value

    def endowment(self, age, gender, startOffset, endOffset, interest,
                  deferral):  # deferral is an average: 0 immediate, 0.5 eoy. Slightly inaccurate for non integer terms
        if interest < self.interest1:
            i_index = 0
        elif interest >= self.interest2 - self.interestStep * 2:
            fake_interest = self.interest2 - self.interestStep * 2
            i_index = int((fake_interest - self.interest1) / self.interestStep)
        else:
            i_index = int((interest - self.interest1) / self.interestStep)
        i1 = i_index * self.interestStep + self.interest1
        i2 = i1 + self.interestStep
        ifrac = (interest - i1) / (i2 - i1)
        table1 = self.allTables[i_index]
        table2 = self.allTables[i_index + 1]
        ageStart = age + startOffset
        startEndowment_1 = table1.endow(ageStart, gender)
        startEndowment_2 = table2.endow(ageStart, gender)
        ageEnd = age + endOffset
        endEndowment_1 = table1.endow(ageEnd, gender)
        endEndowment_2 = table2.endow(ageEnd, gender)
        startEndowment = startEndowment_1 * (1 - ifrac) + startEndowment_2 * ifrac
        endEndowment = endEndowment_1 * (1 - ifrac) + endEndowment_2 * ifrac
        startDiscount = (1 + interest) ** (-startOffset) * table1.tpx(startOffset, age, gender)
        endDiscount = (1 + interest) ** (-endOffset) * table1.tpx(endOffset, age, gender)
        total_value = startDiscount * startEndowment - endDiscount * endEndowment
        total_value = total_value * (1 + interest) ** (0.5 - deferral)
        return total_value

    def endowmentass(self, age, gender, startOffset, endOffset, interest,
                     deferral):  # deferral is an average: 0 immediate, 0.5 eoy. Slightly inaccurate for non integer terms
        if interest < self.interest1:
            i_index = 0
        elif interest >= self.interest2 - self.interestStep * 2:
            fake_interest = self.interest2 - self.interestStep * 2
            i_index = int((fake_interest - self.interest1) / self.interestStep)
        else:
            i_index = int((interest - self.interest1) / self.interestStep)
        i1 = i_index * self.interestStep + self.interest1
        i2 = i1 + self.interestStep
        ifrac = (interest - i1) / (i2 - i1)
        table1 = self.allTables[i_index]
        table2 = self.allTables[i_index + 1]
        ageStart = age + startOffset
        startAssurance_1 = table1.asscap(ageStart, gender)
        startAssurance_2 = table2.asscap(ageStart, gender)
        startEndowment_1 = table1.endow(ageStart, gender)
        startEndowment_2 = table2.endow(ageStart, gender)
        ageEnd = age + endOffset
        endAssurance_1 = table1.asscap(ageEnd, gender)
        endAssurance_2 = table2.asscap(ageEnd, gender)
        endEndowment_1 = table1.endow(ageEnd, gender)
        endEndowment_2 = table2.endow(ageEnd, gender)

        startEndowmentass = (startAssurance_1 + startEndowment_1) * (1 - ifrac) + (
                startAssurance_2 + startEndowment_2) * ifrac
        if endOffset == endOffset - 1:
            endEndowmentass = endAssurance_1 * (1 - ifrac) + endAssurance_2 * ifrac
        else:
            endEndowmentass = (endAssurance_1 + endEndowment_1) * (1 - ifrac) + (
                    endAssurance_2 + endEndowment_2) * ifrac

        startDiscount = (1 + interest) ** (-startOffset) * table1.tpx(startOffset, age, gender)
        endDiscount = (1 + interest) ** (-endOffset) * table1.tpx(endOffset, age, gender)
        total_value = startDiscount * startEndowmentass - endDiscount * endEndowmentass
        total_value = total_value * (1 + interest) ** (0.5 - deferral)
        return total_value

    def Assurance_FirstDeath(self, age, gender, startOffset, endOffset, interest,
                             deferral):  # deferral is an average: 0 immediate, 0.5 eoy. Slightly inaccurate for non integer terms
        if interest < self.interest1:
            i_index = 0
        elif interest >= self.interest2 - self.interestStep * 2:
            fake_interest = self.interest2 - self.interestStep * 2
            i_index = int((fake_interest - self.interest1) / self.interestStep)
        else:
            i_index = int((interest - self.interest1) / self.interestStep)
        i1 = i_index * self.interestStep + self.interest1
        i2 = i1 + self.interestStep
        ifrac = (interest - i1) / (i2 - i1)
        table1 = self.allTables[i_index]
        table2 = self.allTables[i_index + 1]
        ageStart = age + startOffset
        startAssurance_FirstDeath_1 = table1.asscap_joint_first(ageStart, gender)
        startAssurance_FirstDeath_2 = table2.asscap_joint_first(ageStart, gender)
        ageEnd = age + endOffset
        EndAssurance_FirstDeath_1 = table1.asscap_joint_first(ageEnd, gender)
        EndAssurance_FirstDeath_2 = table2.asscap_joint_first(ageEnd, gender)
        startAssurance_FirstDeath = startAssurance_FirstDeath_1 * (1 - ifrac) + startAssurance_FirstDeath_2 * ifrac
        EndAssurance_FirstDeath = EndAssurance_FirstDeath_1 * (1 - ifrac) + EndAssurance_FirstDeath_2 * ifrac
        startDiscount = (1 + interest) ** (-startOffset) * table1.tpx(startOffset, age, gender)
        endDiscount = (1 + interest) ** (-endOffset) * table1.tpx(endOffset, age, gender)
        total_value = startDiscount * startAssurance_FirstDeath - endDiscount * EndAssurance_FirstDeath
        total_value = total_value * (1 + interest) ** (0.5 - deferral)
        return total_value

    def Assurance_SecondDeath(self, age, gender, startOffset, endOffset, interest,
                              deferral):  # deferral is an average: 0 immediate, 0.5 eoy. Slightly inaccurate for non integer terms
        if interest < self.interest1:
            i_index = 0
        elif interest >= self.interest2 - self.interestStep * 2:
            fake_interest = self.interest2 - self.interestStep * 2
            i_index = int((fake_interest - self.interest1) / self.interestStep)
        else:
            i_index = int((interest - self.interest1) / self.interestStep)
        i1 = i_index * self.interestStep + self.interest1
        i2 = i1 + self.interestStep
        ifrac = (interest - i1) / (i2 - i1)
        table1 = self.allTables[i_index]
        table2 = self.allTables[i_index + 1]
        ageStart = age + startOffset
        startAssurance_SecondDeath_1 = table1.asscap_joint_second(ageStart, gender)
        startAssurance_SecondDeath_2 = table2.asscap_joint_second(ageStart, gender)
        ageEnd = age + endOffset
        EndAssurance_SecondDeath_1 = table1.asscap_joint_second(ageEnd, gender)
        EndAssurance_SecondDeath_2 = table2.asscap_joint_second(ageEnd, gender)
        startAssurance_SecondDeath = startAssurance_SecondDeath_1 * (1 - ifrac) + startAssurance_SecondDeath_2 * ifrac
        EndAssurance_SecondDeath = EndAssurance_SecondDeath_1 * (1 - ifrac) + EndAssurance_SecondDeath_2 * ifrac
        startDiscount = (1 + interest) ** (-startOffset) * table1.tpx(startOffset, age, gender)
        endDiscount = (1 + interest) ** (-endOffset) * table1.tpx(endOffset, age, gender)
        total_value = startDiscount * startAssurance_SecondDeath - endDiscount * EndAssurance_SecondDeath
        total_value = total_value * (1 + interest) ** (0.5 - deferral)
        return total_value

    def lx(self, x, gender):
        return self.baseTable.lx(x, gender)

    def tpx(self, t, x, gender):
        return self.baseTable.tpx(t, x, gender)

    def tqx(self, t, x, gender):
        return 1 - self.baseTable.tpx(t, x, gender)


class Tperson:
    def __init__(self, name, DoB, gender):
        self.name = name
        self.DoB = date_in_1(DoB)
        self.gender = gender

    def randomDeath(self, lifeTable, basis):  # return actual date of death
        age = basis.calcDate - self.DoB
        iage = int(age)
        age1 = age
        age2 = iage + 1
        while age1 < basis.projDate - basis.calcDate + age:
            q_age = lifeTable.tqx(1, iage, self.gender)
            rnd_death = rnd_exp(q_age)
            if rnd_death < age2 - age1:
                return rnd_death + age1 + self.DoB
            age1 = age2
            age2 = age2 + 1
            iage = int(age1)
        return float("NaN")


class Tbasis():
    def __init__(self, calcDate, projDate, interest, inflation, salary_incs, fwl, equity_mu, equity_sigma, interest_mu,
                 interest_alpha, interest_sigma, inflation_mu, inflation_alpha, inflation_sigma, pthly):
        self.calcDate = date_in(calcDate)
        self.projDate = date_in(projDate)
        self.interest = interest
        self.inflation = inflation
        self.salary_incs = salary_incs
        self.fwl = fwl
        self.equity_mu = equity_mu
        self.equity_sigma = equity_sigma
        self.interest_mu = interest_mu
        self.interest_alpha = interest_alpha
        self.interest_sigma = interest_sigma
        self.inflation_mu = inflation_mu
        self.inflation_alpha = inflation_alpha
        self.inflation_sigma = inflation_sigma
        self.pthly = pthly

    def getIndex(self, date):
        zi = (date - self.calcDate) * self.pthly
        i = int(zi)
        return i


class Teconomy():
    def __init__(self, basis):
        self.basis = basis
        array_size = int((self.basis.projDate - self.basis.calcDate + 2) * self.basis.pthly)
        self.dates = np.zeros([array_size])
        self.zequities = np.zeros([array_size])
        self.zinterest = np.zeros([array_size])
        self.zinflation = np.zeros([array_size])
        i = 0
        self.dates[i] = self.basis.calcDate
        self.zequities[i] = 100
        self.zinterest[i] = self.basis.interest
        self.zinflation[i] = self.basis.inflation
        while self.dates[i] <= self.basis.projDate + 1:
            i = i + 1
            self.dates[i] = self.dates[i - 1] + 1 / basis.pthly
            # the following is wrong - it is the "idiots" Itos lema (clue)
            r1 = np.random.normal(0, 1)
            r2 = np.random.normal(0, 1)
            r3 = np.random.normal(0, 1)
            self.zequities[i] = self.zequities[i - 1] * (1 + basis.equity_mu / basis.pthly + r1 * basis.equity_sigma / math.sqrt(basis.pthly))
            self.zinterest[i] = self.zinterest[i-1] * np.exp(-basis.interest_alpha *  basis.pthly) + basis.interest_mu * (1 - np.exp(-basis.interest_alpha *  basis.pthly)) + basis.interest_sigma * np.exp(-basis.interest_alpha * 0.5 * basis.pthly) * r2
            self.zinflation[i] = self.zinflation[i-1] * np.exp(-basis.inflation_alpha * basis.pthly) + basis.inflation_mu * (1 - np.exp(-basis.inflation_alpha * basis.pthly)) + basis.inflation_sigma * np.exp(-basis.inflation_alpha * 0.5 * basis.pthly) * r3
    def equities(self, date):
        zi = (date - self.dates[0]) / (self.dates[1] - self.dates[0])
        i = int(zi)
        f = zi - i
        return self.zequities[i] * (1 - f) + self.zequities[i + 1] * f

    def interest(self, date):
        zi = (date - self.dates[0]) / (self.dates[1] - self.dates[0])
        i = int(zi)
        f = zi - i
        return self.zinterest[i] * (1 - f) + self.zinterest[i + 1] * f

    def inflation(self, date):
        zi = (date - self.dates[0]) / (self.dates[1] - self.dates[0])
        i = int(zi)
        f = zi - i
        return self.zinflation[i] * (1 - f) + self.zinflation[i + 1] * f



class Tpolicyholder(Tperson):
    def __init__(self, name, DoB, gender, DoI, pthly):
        Tperson.__init__(self, name, DoB, gender)
        self.DoI = date_in_1(DoI)
        self.pthly = pthly

    def PV(self, economy, lifeTable, date):
        print("Attempting to calculate PV for abstract class")
        return 0

    def cashflows(self, economy, lifeTable, DoDeath):
        print("Attempting to calculate cashflows for abstract class")
        array_size = int((economy.basis.projDate + 2 - economy.basis.calcDate) * economy.basis.pthly)
        cf = np.zeros([array_size])
        return cf


class Tassured(Tpolicyholder):
    def __init__(self, name, DoB, gender, DoI, SA, premium, pthly, term):
        Tpolicyholder.__init__(self, name, DoB, gender, DoI, pthly)
        self.premium = premium
        self.SA = SA
        self.term = term
        self.lapse_rate_first_year = 0.05 # 20% lapse rate in the first year
        self.lapse_rate_subsequent_years = 0.02 # 10% lapse rate in subsequent years

    def is_lapsed(self, basis):
        if self.DoI + self.term > basis.calcDate:

            # Calculate the policy duration
            policy_duration = basis.calcDate - self.DoI
            # If policy has not started yet, it cannot lapse
            if policy_duration < 0:
                # Check if the policy starts before the end of the projection period
                if self.DoI < basis.projDate:
                    diff = basis.projDate - self.DoI
                    policy_duration = 0.0
                    while policy_duration + self.DoI <= basis.projDate and  policy_duration < diff:
                        if policy_duration <= 1:
                            if rnd_poisson(self.lapse_rate_first_year) == 1:
                                return startDate(self.DoI + policy_duration - 0.00001, self.DoI, self.term, self.pthly)
                        else:
                            if rnd_poisson(self.lapse_rate_subsequent_years) == 1:
                                return startDate(self.DoI + policy_duration - 0.00001, self.DoI, self.term, self.pthly)

                        policy_duration += 1 / self.pthly

                else:
                    # If the policy starts after the projection period, it cannot lapse within the projection period
                    return float("NaN")
            else:
                policy_duration = policy_duration
            while policy_duration + self.DoI <= basis.projDate:
                if policy_duration > 1:

                    if rnd_poisson(self.lapse_rate_subsequent_years) == 1:
                        return startDate(self.DoI + policy_duration - 0.00001, self.DoI, self.term, self.pthly)
                else:
                    if rnd_poisson(self.lapse_rate_first_year) == 1:
                        return startDate(self.DoI + policy_duration - 0.00001, self.DoI, self.term, self.pthly)
                policy_duration += 1/self.pthly


            return float("NaN")
        else:
            return float("NaN")

    def PV_out(self, economy, superTable, date, DoDeath , DoLapse):
        if afterDeath(date, DoDeath) or afterlapse(date, DoLapse):
            return 0
        else:
            age = date - self.DoB
            startOffset = max([0, self.DoI - date])
            if np.isnan(self.term):
                endOffset = 100
            else:
                endOffset = self.DoI - date + self.term
            if endOffset < 0:
                return 0
            interest = economy.interest(date)
            deferral = 0
            x = superTable.assurance(age, self.gender, startOffset, endOffset, interest, deferral)
            x = x * self.SA
            return x

    def PV_in(self, economy, superTable, date, DoDeath , DoLapse):
        if afterDeath(date, DoDeath) or afterlapse(date, DoLapse) :
            return 0
        else:
            age = date - self.DoB
            startOffset = startDate(date, self.DoI, self.term, self.pthly) - date
            endOffset = endDate(date, self.DoI, self.term, self.pthly) - date
            if np.isnan(startOffset) or np.isnan(endOffset):
                return 0
            interest = economy.interest(date)
            x = superTable.annuity(age, self.gender, startOffset, endOffset, interest, interest, self.pthly)
            x = x * self.premium * self.pthly
            return x

    def PV(self, economy, superTable, date, DoDeath, DoLapse):
        return self.PV_in(economy, superTable, date, DoDeath, DoLapse) - self.PV_out(economy, superTable, date, DoDeath, DoLapse)

    def cashflows(self, economy, superTable, DoDeath, DoLapse):
        array_size = int((economy.basis.projDate + 2 - economy.basis.calcDate) * economy.basis.pthly)
        cf = np.zeros([array_size])
        premDate = startDate(economy.basis.calcDate, self.DoI, self.term, self.pthly)
        endTermdate = endDate(economy.basis.calcDate, self.DoI, self.term, self.pthly)
        lastDate = min([minDeath(endTermdate,DoDeath), economy.basis.projDate,minLapse(endTermdate, DoLapse)])
        if np.isnan(premDate) or np.isnan(lastDate):
            return cf
        if afterDeath(min([endTermdate, economy.basis.projDate]), DoDeath):
            cf[economy.basis.getIndex(DoDeath)] = -self.SA
        if afterlapse(min([endTermdate, economy.basis.projDate]), DoLapse):
            cf[economy.basis.getIndex(DoLapse)] = 0
        while premDate < lastDate:
            i = economy.basis.getIndex(premDate)
            cf[i] = cf[i] + self.premium
            premDate = premDate + 1 / self.pthly
        return cf


class Tannuitant(Tpolicyholder):
    def __init__(self, name, DoB, gender, DoI, first_payment, pthly, term, increases):
        Tpolicyholder.__init__(self, name, DoB, gender, DoI, pthly)
        self.first_payment = first_payment
        self.term = term
        if np.isnan(increases):
            self.increases = 0
        else:
            self.increases = increases

    def PV_in(self, economy, superTable, date, DoDeath, unused_argument=None):
        return 0

    def PV_out(self, economy, superTable, date, DoDeath, unused_argument=None):
        if afterDeath(date, DoDeath):
            return 0
        else:
            age = date - self.DoB
            startOffset = startDate(date, self.DoI, self.term, self.pthly) - date
            endOffset = endDate(date, self.DoI, self.term, self.pthly) - date
            if np.isnan(startOffset) or np.isnan(endOffset):
                return 0
            interest = economy.interest(date)
            net_interest = interest - self.increases
            x = superTable.annuity(age, self.gender, startOffset, endOffset, interest, net_interest, self.pthly)
            x = x * self.first_payment * paymentIncrement(date + startOffset, self.DoI, self.pthly,
                                                          self.increases) * self.pthly
            return x

    def PV(self, economy, superTable, date, DoDeath, unused_argument=None):
        return self.PV_in(economy, superTable, date, DoDeath) - self.PV_out(economy, superTable, date, DoDeath)

    def cashflows(self, economy, superTable, DoDeath, unused_argument=None):
        array_size = int((economy.basis.projDate + 2 - economy.basis.calcDate) * economy.basis.pthly)
        cf = np.zeros([array_size])
        paymentDate = startDate(economy.basis.calcDate, self.DoI, self.term, self.pthly)
        next_payment = self.first_payment * paymentIncrement(paymentDate, self.DoI, self.pthly, self.increases)
        lastDate = min([minDeath(endDate(economy.basis.calcDate, self.DoI, self.term, self.pthly), DoDeath),
                        economy.basis.projDate])
        if not (np.isnan(self.term)):
            lastDate = min(lastDate, self.DoI + self.term)
        while paymentDate < lastDate:
            i = economy.basis.getIndex(paymentDate)
            cf[i] = cf[i] + next_payment
            paymentDate = paymentDate + 1 / self.pthly
            next_payment = next_payment * (1 + self.increases) ** (1 / self.pthly)
        return cf

class Tendowment(Tpolicyholder):
    def __init__(self, name, DoB, gender, DoI, SA, premium, pthly, term, gteed_amt):
        Tpolicyholder.__init__(self, name, DoB, gender, DoI, pthly)
        self.SA = SA
        self.term = term
        self.premium = premium
        self.gteed_amt = gteed_amt

    def PV_out(self, economy, superTable, date, DoDeath, unused_argument=None):
        age = date - self.DoB
        startOffset = max([0, self.DoI - date])
        if np.isnan(self.term):
            endOffset = 100
        else:
            endOffset = self.DoI - date + self.term
        if endOffset < 0:
            return 0
        interest = economy.interest(date)
        deferral = 0
        x = superTable.endowment(age, self.gender, startOffset, endOffset, interest, deferral)
        x = x * self.gteed_amt
        return x

    def PV_in(self, economy, superTable, date, DoDeath, unused_argument=None):
        if afterDeath(date, DoDeath):
            return 0
        else:
            age = date - self.DoB
            startOffset = startDate(date, self.DoI, self.term, self.pthly) - date
            endOffset = endDate(date, self.DoI, self.term, self.pthly) - date
            if np.isnan(startOffset) or np.isnan(endOffset):
                return 0
            interest = economy.interest(date)
            x = superTable.annuity(age, self.gender, startOffset, endOffset, interest, interest, self.pthly)
            x = x * self.premium * self.pthly
            return x

    def PV(self, economy, superTable, date, DoDeath, unused_argument=None):
        return self.PV_in(economy, superTable, date, DoDeath) - self.PV_out(economy, superTable, date, DoDeath)

    def cashflows(self, economy, superTable, DoDeath, unused_argument=None):  # economy contains a basis which has start and end and pthly. DoDeath is from the DeathMatrix
        array_size = int((economy.basis.projDate + 2 - economy.basis.calcDate) * economy.basis.pthly)
        cf = np.zeros([array_size])
        premDate = startDate(economy.basis.calcDate, self.DoI, self.term, self.pthly)
        endTermDate = endDate(economy.basis.calcDate, self.DoI, self.term, self.pthly)
        lastDate = min([minDeath(endTermDate, DoDeath), economy.basis.projDate])
        if np.isnan(premDate) or np.isnan(lastDate):
            return cf  # returns array of zeros
        if afterDeath(max([economy.basis.projDate, endTermDate]), DoDeath):
            cf[economy.basis.getIndex(DoDeath)] = -self.gteed_amt
        while premDate < lastDate:
            i = economy.basis.getIndex(premDate)
            cf[i] = cf[i] + self.premium
            premDate = premDate + 1 / self.pthly
        return cf

class Tendowmentass(Tpolicyholder):
    def __init__(self, name, DoB, gender, DoI, SA, premium, pthly, term, gteed_amt):
        Tpolicyholder.__init__(self, name, DoB, gender, DoI, pthly)
        self.SA = SA
        self.term = term
        self.premium = premium
        self.gteed_amt = gteed_amt

    def PV_out(self, economy, superTable, date, DoDeath,unused_argument=None):
        age = date - self.DoB
        startOffset = max([0, self.DoI - date])
        if np.isnan(self.term):
            endOffset = 100
        else:
            endOffset = self.DoI - date + self.term
        if endOffset < 0:
            return 0
        interest = economy.interest(date)
        deferral = 0
        x = superTable.endowmentass(age, self.gender, startOffset, endOffset, interest, deferral)
        x = x * self.SA
        return x

    def PV_in(self, economy, superTable, date, DoDeath, unused_argument=None):
        if afterDeath(date, DoDeath):
            return 0
        else:
            age = date - self.DoB
            startOffset = startDate(date, self.DoI, self.term, self.pthly) - date
            endOffset = endDate(date, self.DoI, self.term, self.pthly) - date
            if np.isnan(startOffset) or np.isnan(endOffset):
                return 0
            interest = economy.interest(date)
            x = superTable.annuity(age, self.gender, startOffset, endOffset, interest, interest, self.pthly)
            x = x * self.premium * self.pthly
            return x

    def PV(self, economy, superTable, date, DoDeath, unused_argument=None):
        return self.PV_in(economy, superTable, date, DoDeath) - self.PV_out(economy, superTable, date, DoDeath)

    def cashflows(self, economy, superTable, DoDeath, unused_argument=None):  # economy contains a basis which has start and end and pthly. DoDeath is from the DeathMatrix
        array_size = int((economy.basis.projDate + 2 - economy.basis.calcDate) * economy.basis.pthly)
        cf = np.zeros([array_size])
        premDate = startDate(economy.basis.calcDate, self.DoI, self.term, self.pthly)
        endTermDate = endDate(economy.basis.calcDate, self.DoI, self.term, self.pthly)
        lastDate = min([minDeath(endTermDate, DoDeath), economy.basis.projDate])
        if np.isnan(premDate) or np.isnan(lastDate):
            return cf  # returns array of zeros
        if afterDeath(max([economy.basis.projDate, endTermDate]), DoDeath):
            cf[economy.basis.getIndex(DoDeath)] = -self.gteed_amt
        while premDate < lastDate:
            i = economy.basis.getIndex(premDate)
            cf[i] = cf[i] + self.premium
            premDate = premDate + 1 / self.pthly
        return cf

'''class Tassured_joint_first(Tpolicyholder):
    def __init__(self, name, DoB, gender, DoI, SA, premium, pthly, term, Spouse_DoB):
        Tpolicyholder.__init__(self, name, DoB, gender, DoI, pthly)
        self.SA = SA
        self.term = term
        self.premium = premium
        self.Spouse_DoB = Spouse_DoB

    def PV_out(self, economy, superTable, date, DoDeath, unused_argument=None):
#        if afterDeath(date, DoDeath) :
#            return 0
#        else:
            age_x = date - self.DoB
            age_y = date - self.Spouse_DoB
            startOffset = max([0, self.DoI - date])
            if np.isnan(self.term):
                endOffset = 100
            else:
                endOffset = self.DoI - date + self.term
            if endOffset < 0:
                return 0
            interest = economy.interest(date)
            deferral = 0
            x = superTable.asscap_joint_first(age_x, age_y, self.gender, startOffset, endOffset, interest, deferral)
            x = x * self.SA
            return x

    def PV_in(self, economy, superTable, date, DoDeath, unused_argument=None):
#        if afterDeath(date, DoDeath) :
#            return 0
#        else:
#            age_x = date - self.DoB
#            age_y = date - self.Spouse_DoB
#            startOffset = startDate(date, self.DoI, self.term, self.pthly) - date
#            endOffset = endDate(date, self.DoI, self.term, self.pthly) - date
#            if np.isnan(startOffset) or np.isnan(endOffset):
#                return 0
#            interest = economy.interest(date)
            x = superTable.annuity(age_x, age_y, self.gender, startOffset, endOffset, interest, interest, self.pthly)
            x = x * self.premium * self.pthly
            return x

    def PV(self, economy, superTable, date, DoDeath,  unused_argument=None):
        return self.PV_in(self, economy, superTable, date, DoDeath) - self.PV_out(self, economy, superTable, date, DoDeath)

    def cashflows(self, economy, superTable, DoDeath, unused_argument=None):  # economy contains a basis which has start and end and pthly. DoDeath is from the DeathMatrix
        array_size = int((economy.basis.projDate + 2 - economy.basis.calcDate) * economy.basis.pthly)
        cf = np.zeros([array_size])
        premDate = startDate(economy.basis.calcDate, self.DoI, self.term, self.pthly)
        endTermDate = endDate(economy.basis.calcDate, self.DoI, self.term, self.pthly)
        lastDate = min([minDeath(endTermDate, (min([DoDeath], 10000000))), economy.basis.projDate])
        if np.isnan(premDate) or np.isnan(lastDate):
            return cf  # returns array of zeros
        if afterDeath(min([economy.basis.projDate, endTermDate]), (min([DoDeath], 10000000))):
            cf[economy.basis.getIndex(min([DoDeath], 10000000))] = -self.SA
        while premDate < lastDate:
            i = economy.basis.getIndex(premDate)
            cf[i] = cf[i] + self.premium
            premDate = premDate + 1 / self.pthly
        return cf



class Tassured_joint_second(Tpolicyholder):
    def __init__(self, name, DoB, gender, DoI, SA, premium, pthly, term, Spouse_DoB):
        Tpolicyholder.__init__(self, name, DoB, gender, DoI, pthly)
        self.SA = SA
        self.term = term
        self.premium = premium
        self.Spouse_DoB = Spouse_DoB

    def PV_out(self, economy, superTable, date, DoDeath, unused_argument=None):
#        if afterDeath(date, DoDeath) :
#            return 0
#        else:
            age_x = date - self.DoB
            age_y = date - self.Spouse_DoB
            startOffset = max([0, self.DoI - date])
            if np.isnan(self.term):
                endOffset = 100
            else:
                endOffset = self.DoI - date + self.term
            if endOffset < 0:
                return 0
            interest = economy.interest(date)
            deferral = 0
            x = superTable.asscap_joint_second(age_x,age_y, self.gender, startOffset, endOffset, interest, deferral)
            x = x * self.SA
            return x

    def PV_in(self, economy, superTable, date, DoDeath, unused_argument=None):
#        if afterDeath(date, DoDeath) :
#            return 0
#        else:
            age_x = date - self.DoB
            age_y = date - self.Spouse_DoB
            startOffset = startDate(date, self.DoI, self.term, self.pthly) - date
            endOffset = endDate(date, self.DoI, self.term, self.pthly) - date
            if np.isnan(startOffset) or np.isnan(endOffset):
                return 0
            interest = economy.interest(date)
            x = superTable.annuity(age_x,age_y, self.gender, startOffset, endOffset, interest, interest, self.pthly)
            x = x * self.premium * self.pthly
            return x

    def PV(self, economy, superTable, date, DoDeath, unused_argument=None):
        return self.PV_in(self, economy, superTable, date, DoDeath) - self.PV_out(self, economy, superTable, date, DoDeath)

    def cashflows(self, economy, superTable, DoDeath, unused_argument=None):  # economy contains a basis which has start and end and pthly. DoDeath is from the DeathMatrix
        array_size = int((economy.basis.projDate + 2 - economy.basis.calcDate) * economy.basis.pthly)
        cf = np.zeros([array_size])
        premDate = startDate(economy.basis.calcDate, self.DoI, self.term, self.pthly)
        endTermDate = endDate(economy.basis.calcDate, self.DoI, self.term, self.pthly)
        lastDate = min([minDeath(endTermDate, (max([DoDeath], 1000000))), economy.basis.projDate])
        if np.isnan(premDate) or np.isnan(lastDate):
            return cf  # returns array of zeros
        if afterDeath(min([economy.basis.projDate, endTermDate]), (max([DoDeath], 100000))):
            cf[economy.basis.getIndex(max([DoDeath], 100000))] = -self.SA
        while premDate < lastDate:
            i = economy.basis.getIndex(premDate)
            cf[i] = cf[i] + self.premium
            premDate = premDate + 1 / self.pthly
        return cf'''





class TbalanceSheet():
    def __init__(self, cash, equities, bonds, bond_duration, asatDate):
        self.cash = cash
        self.equities = equities
        self.bonds = bonds
        self.bond_duration = bond_duration
        self.asatDate = asatDate

def create_balance_sheet(total_assets, bond_duration,asatDate ):
    cash_percentage = 10
    equities_percentage = 30
    bonds_percentage = 60


    cash = total_assets * (cash_percentage / 100)
    equities = total_assets * (equities_percentage / 100)
    bonds = total_assets * (bonds_percentage / 100)

    return TbalanceSheet(cash, equities, bonds, bond_duration, asatDate)


class TLifeCompany():
    def __init__(self, balanceSheet, rent, salaries, policyholders):
        self.balanceSheet = balanceSheet
        self.rent = rent
        self.salaries = salaries
        self.policyholders = policyholders

    def valuation(self, economy, lifeTable, data):
        total_val = 0
        for ph in self.policyholders:
            val = Tpolicyholder(ph).PV(economy, lifeTable, data)
            total_val = total_val + val
        return total_val



    def rollForward(self, lastBS, economy, cashflows, index):
        t = 1 / economy.basis.pthly
        date1 = economy.basis.calcDate + index * t
        date2 = date1 + t
        date12 = (date1 + date2) / 2
        av_int = economy.interest(date12)
        newCash = lastBS.cash * (1 * av_int) ** t - cashflows[index] * (1 + av_int) ** (t / 2)
        newCash = newCash - (self.rent + self.salaries) * t
        newEquities = lastBS.equities * economy.equities(date2) / economy.equities(date1)
        newBonds = lastBS.bonds * (1 + av_int) ** t
        # Rebalance if cash is below zero or if it's the first day of the year
        if newCash < 0 or date2 in (2025.0, 2026.0, 2027.0):
            total_assets = newCash + newEquities + newBonds
            newCash = total_assets * 0.10
            newEquities = total_assets * 0.30
            newBonds = total_assets * 0.60
        int1 = economy.interest(date1)
        int2 = economy.interest(date2)
        d = self.balanceSheet.bond_duration
        newBonds = newBonds * ((1 + int1) ** d) / ((1 + int2) ** d)
        newBS = TbalanceSheet(newCash, newEquities, newBonds, d, date2)
        return newBS


def paymentIncrement(next_prem_date, DoI, pthly, increases):
    payment_index = math.ceil((next_prem_date - DoI) * pthly)
    increment = (1 + increases) ** (payment_index / pthly)
    return increment

def rnd_poisson(lam):
    # lam is the average rate (lambda) of events
    L = math.exp(-lam)
    k = 0
    p = 1
    while p > L:
        k += 1
        p *= random.random()
    return k - 1





def days_in_month(m, y):
    if m == 2:
        if y % 4 == 0:
            dim = 29
        else:
            dim = 28
    elif (m == 4) or (m == 6) or (m == 9) or (m == 11):
        dim = 30
    else:
        dim = 31
    return dim


def date_in(s):
    if type(s) == str:
        a = s.split("/")
        d = int(a[0])
        m = int(a[1])
        y = int(a[2])
        rdate = y + (m - 1) / 12 + (d - 1) / (12 * days_in_month(m, y))
    else:
        rdate = 0
    return rdate


def date_in_1(s):
    if type(s) == str:
        a = s.split("-")
        d = int(a[0])
        m = int(a[1])
        y = int(a[2])
        rdate = y + (m - 1) / 12 + (d - 1) / (12 * days_in_month(m, y))
    else:
        rdate = 0
    return rdate


def date_out(rdate):
    y = int(rdate)
    m_frac = rdate - y
    m = int((0.0001 + m_frac * 12)) + 1
    d_frac = rdate - y - (m - 1) / 12
    d = int(0.0001 + d_frac * 12 * days_in_month(m, y) + 1)
    s = datetime.datetime(y, m, d)
    return s


def afterDeath(date, DoDeath):
    if np.isnan(DoDeath):
        return False
    elif DoDeath == 0:
        return False
    else:
        return date > DoDeath

def afterlapse(date, DoLapse):
    if np.isnan(DoLapse):
        return False
    elif DoLapse == 0:
        return False
    else:
        return date > DoLapse


def minDeath(date, DoDeath):
    if np.isnan(DoDeath):
        return date
    elif DoDeath == 0:
        return date
    else:
        return min([date, DoDeath])

def minLapse(date, DoLapse):
    if np.isnan(DoLapse):
        return date
    elif DoLapse == 0:
        return date
    else:
        return min([date, DoLapse])


def startDate(date, DOI, term, pthly):  # date when we make the next payment
    if np.isnan(term):
        use_term = 100
    else:
        use_term = term

    if DOI + use_term < date:
        return float("NAN")
    elif date > DOI:
        n_pths = (date - DOI) * pthly
        frac = 1 - (n_pths - int(n_pths))  # time left for next payment
        return date + frac / pthly
    else:
        return DOI


def endDate(date, DoI, term, pthly):
    if np.isnan(term):
        return date + 100
    if DoI + term < date:
        return float("Nan")
    else:
        return DoI + term


def rnd_exp(q):
    mu = -math.log(1 - q)
    y = random.random()
    t = -math.log(y) / mu
    return t


def read_policyholders(csvFile):
    policyHolderData = pd.read_csv(csvFile)
    phList = []
    for iph in policyHolderData.index:
        ph = policyHolderData.loc[iph]
        if ph.loc["policy type"] == "term":
            new_ph = Tassured(ph.loc["name"], ph.loc["DoB"], ph.loc["gender"], ph.loc["DoI"], ph.loc["SA"],
                              ph.loc["premium"], ph.loc["pthly"], ph.loc["term"])
        elif ph.loc["policy type"] == "whole life":
            new_ph = Tassured(ph.loc["name"], ph.loc["DoB"], ph.loc["gender"], ph.loc["DoI"], ph.loc["SA"],
                              ph.loc["premium"], ph.loc["pthly"], ph.loc["term"])
        elif ph.loc["policy type"] == "annuity constant":
            new_ph = Tannuitant(ph.loc["name"], ph.loc["DoB"], ph.loc["gender"], ph.loc["DoI"], ph.loc["first_payment"],
                                ph.loc["pthly"], ph.loc["term"], 0)
        elif ph.loc["policy type"] == "annuity fixed":
            new_ph = Tannuitant(ph.loc["name"], ph.loc["DoB"], ph.loc["gender"], ph.loc["DoI"], ph.loc["first_payment"],
                                ph.loc["pthly"], ph.loc["term"], ph.loc["increases"])
        elif ph.loc["policy type"] == "endowment" or ph.loc["policy type"] == "endowment assurance":
            new_ph = Tendowment(ph.loc["name"], ph.loc["DoB"], ph.loc["gender"], ph.loc["DoI"], ph.loc["SA"],
                                ph.loc["premium"], ph.loc["pthly"], ph.loc["term"], ph.loc["gteed amount"])
#        elif ph.loc["policy type"] == "annuity fixed + spouse":
#            new_ph = Tannuitant(ph.loc["name"], ph.loc["DoB"], ph.loc["gender"], ph.loc["DoI"], ph.loc["first payment"],
#                                ph.loc["pthly"], ph.loc["term"], ph.loc["increases"],ph.loc["Spouse DoB"],ph.loc["Spouse prop"])
#        elif ph.loc["policy type"] == "whole life first death":
#            new_ph = Tassured_joint_first(ph.loc["name"], ph.loc["DoB"], ph.loc["gender"], ph.loc["DoI"], ph.loc["SA"],
#                                          ph.loc["premium"], ph.loc["pthly"], float("NaN"), ph.loc["Spouse DoB"])
#        elif ph.loc["policy type"] == "term first death":
#            new_ph = Tassured_joint_first(ph.loc["name"], ph.loc["DoB"], ph.loc["gender"], ph.loc["DoI"], ph.loc["SA"],
#                                          ph.loc["premium"], ph.loc["pthly"], ph.loc["term"], ph.loc["Spouse DoB"])
#        elif ph.loc["policy type"] == "whole life second death":
#            new_ph = Tassured_joint_second(ph.loc["name"], ph.loc["DoB"], ph.loc["gender"], ph.loc["DoI"], ph.loc["SA"],
#                                           ph.loc["premium"], ph.loc["pthly"], float("NaN"), ph.loc["Spouse DoB"])
#        elif ph.loc["policy type"] == "term second death":
#            new_ph = Tassured_joint_second(ph.loc["name"], ph.loc["DoB"], ph.loc["gender"], ph.loc["DoI"], ph.loc["SA"],
#                                           ph.loc["premium"], ph.loc["pthly"], ph.loc["term"], ph.loc["Spouse DoB"])
        else:
            new_ph = None
        if new_ph != None:
            phList.append(new_ph)
    return phList




print(time1 := datetime.datetime.now())
UK2019 = TsuperTable("Actuarial Table for 2018-2020", "..\lifeTable.xlsx", -0.1, 0.25, 0.001)
print("Annuities and Assurances are now set up")
policyholderList = read_policyholders("..\policyHolders.csv")
print("Setting up main basis")
mainBasis = Tbasis('1/1/2024', '1/1/2027',
                   0.04, 0.06, 0.08, 12, 0.08, 0.16, 0.05,
                   0.5, 0.01, 0.02, 0.5, 0.01, 12)
print("Running Economic Projections")
allEconomies = []
total_MC = 1000
deathMatrix = np.zeros([len(policyholderList), total_MC])
for i in range(total_MC):
    economy_run = Teconomy(mainBasis)
    allEconomies.append(economy_run)
    for iph in range(len(policyholderList)):
        ph = policyholderList[iph]
        DoD = ph.randomDeath(UK2019, mainBasis)
        deathMatrix[iph, i] = DoD

'''num_policyholders_with_spouses = sum(1 for ph in policyholderList if ph["policy type"] in ["whole life first death", "whole life second death", "term first death", "term second death", "annuity fixed + spouse"])
spouse_deathMatrix = np.zeros([num_policyholders_with_spouses, total_MC])

for iph, ph in enumerate(policyholderList):
    if ph["policy type"] in ["whole life first death", "whole life second death", "term first death", "term second death", "annuity fixed + spouse"]:
        for i in range(total_MC):
            Spouse_DoD = ph.randomDeath(UK2019, mainBasis)
            if np.isnan(Spouse_DoD):
                spouse_deathMatrix[iph, i] = float('Nan')
            else:
                spouse_deathMatrix[iph, i] = Spouse_DoD'''

lapseMatrix = np.zeros([len(policyholderList), total_MC])
for i in range(total_MC):
    economy_run = Teconomy(mainBasis)
    allEconomies.append(economy_run)
    for iph in range(len(policyholderList)):
        ph = policyholderList[iph]
        # Add the condition here
        if isinstance(ph, Tassured):
            lapse = ph.is_lapsed(mainBasis)
            lapseMatrix[iph, i] = lapse
        else:
            # If ph is not an instance of TAssured, assign NaN to the lapseMatrix
            lapseMatrix[iph, i] = float("NaN")



array_size = int((mainBasis.projDate + 2 - mainBasis.calcDate) * mainBasis.pthly)
balanceSheet = create_balance_sheet(8300000,7,mainBasis.calcDate)
#balanceSheet = TbalanceSheet(1000000, 1000000, 4000000, 7, mainBasis.calcDate)
company = TLifeCompany(balanceSheet, 50000, 800000, policyholderList)
total_liability = np.zeros([total_MC])
total_cashflows = np.zeros([total_MC, array_size])
for isim in range(total_MC):
    for iph in range(len(policyholderList)):
        ph = policyholderList[iph]
        economy = allEconomies[isim]
        total_liability[isim] = total_liability[isim] + ph.PV(economy, UK2019, mainBasis.projDate, deathMatrix[iph, isim],  lapseMatrix[iph, isim] )
        add_cashflows = ph.cashflows(economy, UK2019, deathMatrix[iph, isim],  lapseMatrix[iph, isim])
        total_cashflows[isim, :] = np.add(total_cashflows[isim, :], add_cashflows)



BSMatrix = []
for isim in range(total_MC):
    BSMatrix.append([])
    BSMatrix[isim].append(company.balanceSheet)
    previousBS = company.balanceSheet
    index = 0
    economy = allEconomies[isim]
    cashflows = total_cashflows[isim, :]
    while mainBasis.calcDate + index / mainBasis.pthly <= mainBasis.projDate:
        newBS = company.rollForward(previousBS, economy, cashflows, index)
        BSMatrix[isim].append(newBS)
        previousBS = newBS
        index = index + 1

print("Balance sheet calculated")

solvency = np.zeros([total_MC])
assets = np.zeros([total_MC])
for isim in range(total_MC):
    balanceSheets = BSMatrix[isim]
    index = int((mainBasis.projDate - mainBasis.calcDate) * mainBasis.pthly)
    this_BS = balanceSheets[index]
    assets[isim] = this_BS.cash + this_BS.equities + this_BS.bonds
    solvency[isim] = this_BS.cash + this_BS.equities + this_BS.bonds - total_liability[isim]

present_Value = ph.PV(economy, UK2019, 2024, deathMatrix[iph, isim],  lapseMatrix[iph, isim] )
print(present_Value)

# Sort the solvency array to find the 95th percentile
solvency_sorted = np.sort(solvency)
# Find the 95th percentile index
percentile_index = int(np.ceil(0.95 * total_MC)) - 1
print(percentile_index)
# Get the solvency amount at the 95th percentile
solvency_95th_percentile = solvency_sorted[percentile_index]
print(solvency_95th_percentile)

# This is the total amount of assets required at the start to be solvent in 95% of cases
required_initial_assets = solvency_95th_percentile
print(solvency)
plt.hist(solvency, color="red", bins=60)
plt.title("Solvency")
plt.ylabel("Frequency")
plt.xlabel("Assets - Liabilities")
plt.show()