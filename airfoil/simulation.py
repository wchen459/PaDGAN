#import os
from __future__ import division
import configparser
import pexpect
#import subprocess as sp
import gc

import numpy as np
from scipy.interpolate import interp1d

import sys
sys.path.append('..')
from utils import safe_remove, create_dir


def compute_coeff(airfoil, reynolds=500000, mach=0, alpha=3, n_iter=200, tmp_dir='./tmp'):
    
    create_dir(tmp_dir)
    
    gc.collect()
    safe_remove('{}/airfoil.log'.format(tmp_dir))
    fname = '{}/airfoil.dat'.format(tmp_dir)
    with open(fname, 'wb') as f:
        np.savetxt(f, airfoil)
    
    try:
        # Has error: Floating point exception (core dumped)
        # This is the "empty input file: 'tmp/airfoil.log'" warning in other approaches
        child = pexpect.spawn('xfoil')
        timeout = 10
        
        child.expect('XFOIL   c> ', timeout)
        child.sendline('load tmp/airfoil.dat')
        child.expect('Enter airfoil name   s> ', timeout)
        child.sendline('af')
        child.expect('XFOIL   c> ', timeout)
        child.sendline('OPER')
        child.expect('.OPERi   c> ', timeout)
        child.sendline('VISC {}'.format(reynolds))
        child.expect('.OPERv   c> ', timeout)
        child.sendline('ITER {}'.format(n_iter))
        child.expect('.OPERv   c> ', timeout)
        child.sendline('MACH {}'.format(mach))
        child.expect('.OPERv   c> ', timeout)
        child.sendline('PACC')
        child.expect('Enter  polar save filename  OR  <return> for no file   s> ', timeout)
        child.sendline('{}/airfoil.log'.format(tmp_dir))
        child.expect('Enter  polar dump filename  OR  <return> for no file   s> ', timeout)
        child.sendline()
        child.expect('.OPERva   c> ', timeout)
        child.sendline('ALFA {}'.format(alpha))
        child.expect('.OPERva   c> ', timeout)
        child.sendline()
        child.expect('XFOIL   c> ', timeout)
        child.sendline('quit')
        
        child.expect(pexpect.EOF)
        child.close()
        
        # Has the dead lock issue
#        with open('tmp/control.in', 'w') as text_file:
#            text_file.write('load tmp/airfoil.dat\n' +
#                            'af\n' +
#                            'OPER\n' +
#                            'VISC {}\n'.format(reynolds) +
#                            'ITER {}\n'.format(n_iter) +
#                            'MACH {}\n'.format(mach) +
#                            'PACC\n' +
#                            'tmp/airfoil.log\n' +
#                            '\n' +
#                            'ALFA {}\n'.format(alpha) +
#                            '\n' +
#                            'quit\n')
#        os.system('xfoil <tmp/control.in> tmp/airfoil.out')
        
        # Has the dead lock issue
        # Has memory issue
#        ps = sp.Popen(['xfoil'], stdin=sp.PIPE, stderr=sp.PIPE, stdout=sp.PIPE)
#        
#        # Use communicate() rather than .stdin.write, .stdout.read or .stderr.read 
#        # to avoid deadlocks due to any of the other OS pipe buffers filling up and 
#        # blocking the child process.
#        out, err = ps.communicate('load tmp/airfoil.dat\n' +
#                                  'af\n' +
#                                  'OPER\n' +
#                                  'VISC {}\n'.format(reynolds) +
#                                  'ITER {}\n'.format(n_iter) +
#                                  'MACH {}\n'.format(mach) +
#                                  'PACC\n' +
#                                  'tmp/airfoil.log\n' +
#                                  '\n' +
#                                  'ALFA {}\n'.format(alpha) +
#                                  '\n' +
#                                  'quit\n')
    
        res = np.loadtxt('{}/airfoil.log'.format(tmp_dir), skiprows=12)
        if len(res) == 9:
            CL = res[1]
            CD = res[2]
        else:
            CL = -np.inf
            CD = np.inf
            
    except Exception as ex:
#        print(ex)
        print('XFoil error!')
        CL = -np.inf
        CD = np.inf
        
    safe_remove(':00.bl')
    
    return CL, CD

def read_config(config_fname):
    
    # Airfoil operating conditions
    Config = configparser.ConfigParser()
    Config.read(config_fname)
    reynolds = float(Config.get('OperatingConditions', 'Reynolds'))
    mach = float(Config.get('OperatingConditions', 'Mach'))
    alpha = float(Config.get('OperatingConditions', 'Alpha'))
    n_iter = int(Config.get('OperatingConditions', 'N_iter'))
    
    return reynolds, mach, alpha, n_iter

def detect_intersect(airfoil):
    # Get leading head
    lh_idx = np.argmin(airfoil[:,0])
    lh_x = airfoil[lh_idx, 0]
    # Get trailing head
    th_x = np.minimum(airfoil[0,0], airfoil[-1,0])
    # Interpolate
    f_up = interp1d(airfoil[:lh_idx+1,0], airfoil[:lh_idx+1,1])
    f_low = interp1d(airfoil[lh_idx:,0], airfoil[lh_idx:,1])
    xx = np.linspace(lh_x, th_x, num=1000)
    yy_up = f_up(xx)
    yy_low = f_low(xx)
    # Check if intersect or not
    if np.any(yy_up < yy_low):
        return True
    else:
        return False

def evaluate(airfoil, return_CL_CD=False, config_fname='op_conditions.ini', tmp_dir='./tmp'):

    reynolds, mach, alpha, n_iter = read_config(config_fname)
    CL, CD = compute_coeff(airfoil, reynolds, mach, alpha, n_iter, tmp_dir)
    perf = CL/CD
    if perf < -50 or perf > 220:
        perf = np.nan
    
    if return_CL_CD:
        return perf, CL, CD
    else:
        return perf
    
    
if __name__ == "__main__":
    
#    airfoil = np.load('tmp/a18sm.npy')
#    airfoils = np.load('data/airfoil_interp.npy')
    airfoils = np.load('data/xs_train.npy')
    
    idx = np.random.choice(airfoils.shape[0])
    airfoil = airfoils[idx]
    
    # Read airfoil operating conditions from a config file
    config_fname = 'op_conditions.ini'
    reynolds, mach, alpha, n_iter = read_config(config_fname)
    
    CL, CD = compute_coeff(airfoil, reynolds, mach, alpha, n_iter)
    print(CL/CD, CL, CD)
    print(np.load('data/ys_train.npy')[idx])
    
#    val = evaluate(airfoil, return_CL_CD=False)
#    print(val)
