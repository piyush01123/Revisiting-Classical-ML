from setuptools import setup, find_packages
setup(
	name="RCML",
	version="0.1",
	package_dir={'RCML':'src'},
	packages=['RCML'],
	install_requires=['pandas','numpy','scikit-learn','matplotlib','cvxpy'],
	author="Piyush Singh"
)
