import unittest
import pandas as pd
import numpy as np
from fillna import render

class TestFillNA(unittest.TestCase):
 
	def setUp(self):
		# Test data includes:
		#  - columns of float and string types
		#  - column of ints with zeros (should not be filled)
		#  - columns with first or last values empty
		#  - completely filled and completely empty columns
		self.table = pd.DataFrame([
			['',					2,			3.14,		'a',	None, ''],
			['frederson',	0,			None,		'b', 	None, ''], 
			['', 					-10, 		None, 	'c', 	None, ''],
			['',					-2,			10,			'd', 	None,	''], 
			['maggie',		8,			None,		'e', 	None,	'']], 
			columns=['string','int','float','filledstr','emptyfloat','emptystring'])

		# Pandas should infer these types anyway, but leave nothing to chance
		self.table = self.table.astype({
			'string':'str', 
			'int':'int64',
			'float':'float64',
			'filledstr':'str',
			'emptyfloat':'float64',
			'emptystring':'str'
			})

		self.filleddown = pd.DataFrame([
			['',					2,			3.14,		'a',	None, ''],
			['frederson',	0,			3.14,		'b', 	None, ''], 
			['frederson', -10, 		3.14, 	'c', 	None, ''],
			['frederson',	-2,			10,			'd', 	None,	''], 
			['maggie',		8,			10,			'e', 	None,	'']], 
			columns=['string','int','float','filledstr','emptyfloat','emptystring'])
	
		self.filleddown = self.filleddown.astype({
			'string':'str', 
			'int':'int64',
			'float':'float64',
			'filledstr':'str',
			'emptyfloat':'float64',
			'emptystring':'str'
			})

		self.filledup = pd.DataFrame([
			['frederson',	2,			3.14,		'a',	None, ''],
			['frederson',	0,			10,			'b', 	None, ''], 
			['maggie', 		-10, 		10, 		'c', 	None, ''],
			['maggie',		-2,			10,			'd', 	None,	''], 
			['maggie',		8,			None,			'e', 	None,	'']], 
			columns=['string','int','float','filledstr','emptyfloat','emptystring'])
	
		self.filledup = self.filledup.astype({
			'string':'str', 
			'int':'int64',
			'float':'float64',
			'filledstr':'str',
			'emptyfloat':'float64',
			'emptystring':'str'
			})

	def test_NOP(self):
		params = { 'colnames':'', 'method':0}
		out = render(self.table, params)
		self.assertTrue(out.equals(self.table)) # should NOP when first applied

	def test_fill_down(self):
		params = { 'colnames': 'string,int,float,filledstr,emptyfloat,emptystring', 'method':0}
		out = render(self.table, params)
		self.assertTrue(out.equals(self.filleddown))

	def test_fill_up(self):
		params = { 'colnames': 'string,int,float,filledstr,emptyfloat,emptystring', 'method':1}
		out = render(self.table, params)
		self.assertTrue(out.equals(self.filledup))

	def test_one_column(self):
		params = { 'colnames': 'emptystring', 'method':0}
		out = render(self.table, params)
		self.assertTrue(out.equals(self.table))  # in this case NOP

		params = { 'colnames': 'string', 'method':0}
		out = render(self.table, params)
		self.assertTrue(out['string'].equals(self.filleddown['string'])) # filled this col
		out = out.drop('string', axis=1)
		table_minus_string = self.table.drop('string', axis=1)
		self.assertTrue(out.equals(table_minus_string))										# but not these cols


if __name__ == '__main__':
    unittest.main()


