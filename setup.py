from setuptools import setup, find_packages
setup(
	name="RCML",
	version="1.1",
	package_dir={'RCML':'src'},
	packages=['RCML'],
	install_requires=['pandas','numpy','scikit-learn','matplotlib','cvxpy'],
	license=open('LICENSE').read(),
	description="Minimal implementations of classical ML algorithms",
	long_description=open("README.md").read(),
	long_description_content_type='text/markdown',
	author="Piyush Singh"
)
