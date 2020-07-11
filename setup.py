from setuptools import setup, find_packages

setup(name='MADDPG Congestion Control on OMNET++',
      version='0.0.1',
      description='Requirement',
      url='https://github.com/lqy0057/MADDPG-Congestion-Control-Based-on-OMNETPP',
      author='Quanyi Li',
      author_email='liquanyi@bupt.edu.cn',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=['gym==0.10.0', 'tensorflow=1.2.0','matplotlib']
)
