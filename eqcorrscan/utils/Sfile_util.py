#!/usr/bin/python
"""
Part of the EQcorrscan module to read nordic format s-files and write them
EQcorrscan is a python module designed to run match filter routines for
seismology, within it are routines for integration to seisan and obspy.
With obspy integration (which is necessary) all main waveform formats can be
read in and output.

Code generated by Calum John Chamberlain of Victoria University of Wellington,
2015.

Copyright 2015 Calum Chamberlain

This file is part of EQcorrscan.

    EQcorrscan is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    EQcorrscan is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with EQcorrscan.  If not, see <http://www.gnu.org/licenses/>.

"""


from obspy import UTCDateTime
import numpy as np

class PICK:
    """
    Pick information for seisan implimentation, note all fields can be left\
    blank to obtain a default pick: picks have a print function which will\
    print them as they would be seen in an S-file.

    Attributes:
        :type station: str
        :param station: Station name, less than five charectars required as\
         standard
        :type channel: str
        :param channel: Two or three charactar channel name, stored as two\
            charactars in S-file
        :type impulsivity: str
        :param impulsivity: either 'C' or 'D' for compressive and dilatational
        :type phase: str
        :param phase: Any allowable phase name in two characters
        :type weight: int
        :param weight: 0-4 with 0=100%, 4=0%, use weight=9 for unknown timing
        :type polarity: str
        :type time: obspy.UTCDateTime()
        :param time: Pick time as an obspy.UTCDateTime object
        :type coda: int
        :param coda: Length of coda in seconds
        :type amplitude: float
        :param amplitude: Amplitude (zero-peak), type is given in phase
        :type peri: float
        :param peri: Period of amplitude
        :type azimuth: float
        :param azimuth: Direction of approach in degrees
        :type velocity: float
        :param velocity: Phase velocity (km/s)
        :type AIN: int
        :param AIN: Angle of incidence.
        :type SNR: float
        :param SNR: Signal to noise ratio
        :type azimuthres: int
        :param azimuthres: Residual azimuth
        :type timeres: float
        :param timeres: Time residual in seconds
        :type finalweight: int
        :param finalweight: Final weight used in location
        :type distance: float
        :param distance: Source-reciever distance in km
        :type CAZ: int
        :param CAZ: Azimuth at source.
    """
    pickcount=0
    def __init__(self, station=' ', channel=' ', impulsivity=' ', phase=' ',
                 weight=999, polarity=' ', time=UTCDateTime(0),
                 coda=999, amplitude=float('NaN'),
                 peri=float('NaN'), azimuth=float('NaN'),
                 velocity=float('NaN'), AIN=999, SNR=float('NaN'),
                 azimuthres=999, timeres=float('NaN'),
                 finalweight=999, distance=float('NaN'),
                 CAZ=999):
        self.station=station
        self.channel=channel
        self.impulsivity=impulsivity
        self.phase=phase
        self.weight=weight
        self.polarity=polarity
        self.time=time
        self.coda=coda
        self.amplitude=amplitude
        self.peri=peri
        self.azimuth=azimuth
        self.velocity=velocity
        self.AIN=AIN
        self.SNR=SNR
        self.azimuthres=azimuthres
        self.timeres=timeres
        self.finalweight=finalweight
        self.distance=distance
        self.CAZ=CAZ
        self.pickcount+=1
    def __repr__(self):
        return "PICK()"
    def __str__(self):
        if self.distance >= 100.0:
            self.distance = _int_conv(self.distance)
        elif 10.0 < self.distance < 100.0:
            self.distance = round(self.distance, 1)
            round_len = 1
        elif self.distance < 10.0:
            self.distance = round(self.distance, 2)
            round_len = 2
        else:
            round_len = False
        if self.peri < 10.0:
            peri_round = 2
        elif self.peri >= 10.0:
            peri_round = 1
        else:
            peri_round = False
        if not self.AIN == '':
            if not np.isnan(self.AIN):
                dummy = int(self.AIN)
            else:
                dummy = self.AIN
        else:
            dummy = self.SNR
        print_str = ' ' + self.station.ljust(5) +\
            self.channel[0]+self.channel[len(self.channel)-1] +\
            ' ' + self.impulsivity +\
            self.phase.ljust(4) +\
            _str_conv(self.weight).rjust(1) + ' ' +\
            self.polarity.rjust(1) + ' ' +\
            str(self.time.hour).rjust(2) +\
            str(self.time.minute).rjust(2) +\
            str(self.time.second).rjust(3) + '.' +\
            str(float(self.time.microsecond) /
                (10 ** 4)).split('.')[0].zfill(2) +\
            _str_conv(int(self.coda)).rjust(5)[0:5] +\
            _str_conv(round(self.amplitude, 1)).rjust(7)[0:7] +\
            _str_conv(self.peri, rounded=peri_round).rjust(5) +\
            _str_conv(self.azimuth).rjust(6) +\
            _str_conv(self.velocity).rjust(5) +\
            _str_conv(dummy).rjust(4) +\
            _str_conv(int(self.azimuthres)).rjust(3) +\
            _str_conv(self.timeres, rounded=2).rjust(5) +\
            _str_conv(int(self.finalweight)).rjust(2) +\
            _str_conv(self.distance, rounded=round_len).rjust(5) +\
            _str_conv(int(self.CAZ)).rjust(4)+' '
        return print_str

    def write(self, filename):
        """
        Public function to write the pick to a file

        :type filename: str
        :param filename: Path to file to write to - will append to file
        """
        import os
        import warnings
        if os.path.isfile(filename):
            open_as='a'
        else:
            warnings.warn('File does not exist, no header')
            open_as='w'

        with open(filename,open_as) as f:
            pickstr=self.__str__()
            f.write(pickstr+'\n')
        return


class EVENTINFO:
    """
    Header information for seisan events, again all fields can be left blank for\
    a default empty header.  The print function for header will print important\
    information, but not as seen in an S-file.

    For more information on parameters see the seisan manual.

    Attributes:
        :type time: obspy.UTCDateTime
        :param time: Event origin time
        :type loc_mod_ind: str
        :param loc_mod_ind:
        :type dist_ind: str
        :param dist_ind: Distance flag, usually 'L' for local, 'R' for regional\
            and 'D' for distant
        :type ev_id: str
        :param ev_id: Often blank, 'E' denotes explosion and fixes depth to 0km
        :type latitude: float
        :param latitude: Hypocentre latitude in decimal degrees
        :type longitude: float
        :param lognitude: Hypocentre longitude in decimal degrees
        :type depth: float
        :param depth: hypocentre depth in km
        :type depth_ind: str
        :param depth_ind:
        :type loc_ind: str
        :param loc_ind:
        :type agency: str
        :param agency: Reporting agency, three letters
        :type nsta: int
        :param nsta: Number of stations recording
        :type t_RMS: float
        :param t_RMS: Root-mean-squared time residual
        :type Mag_1: float
        :param Mag_1: first magnitude
        :type Mag_1_type: str
        :param Mag_1_type: Type of magnitude for Mag_1 ('L', 'C', 'W')
        :type Mag_1_agency: str
        :param Mag_1_agency: Reporting agency for Mag_1
        :type Mag_2: float
        :param Mag_2: second magnitude
        :type Mag_2_type: str
        :param Mag_2_type: Type of magnitude for Mag_2 ('L', 'C', 'W')
        :type Mag_2_agency: str
        :param Mag_2_agency: Reporting agency for Mag_2
        :type Mag_3: float
        :param Mag_3: third magnitude
        :type Mag_3_type: str
        :param Mag_3_type: Type of magnitude for Mag_3 ('L', 'C', 'W')
        :type Mag_3_agency: str
        :param Mag_3_agency: Reporting agency for Mag_3
    """
    def __init__(self, time=UTCDateTime(0), loc_mod_ind=' ', dist_ind=' ',
                 ev_id=' ', latitude=float('NaN'), longitude=float('NaN'),
                 depth=float('NaN'), depth_ind=' ', loc_ind=' ', agency=' ',
                 nsta=0, t_RMS=float('NaN'), Mag_1=float('NaN'),
                 Mag_1_type=' ', Mag_1_agency=' ', Mag_2=float('NaN'),
                 Mag_2_type=' ', Mag_2_agency=' ', Mag_3=float('NaN'),
                 Mag_3_type=' ', Mag_3_agency=' '):
        self.time=time
        self.loc_mod_ind=loc_mod_ind
        self.dist_ind=dist_ind
        self.ev_id=ev_id
        self.latitude=latitude
        self.longitude=longitude
        self.depth=depth
        self.depth_ind=depth_ind
        self.loc_ind=loc_ind
        self.agency=agency
        self.nsta=nsta
        self.t_RMS=t_RMS
        self.Mag_1=Mag_1
        self.Mag_1_type=Mag_1_type
        self.Mag_1_agency=Mag_1_agency
        self.Mag_2=Mag_2
        self.Mag_2_type=Mag_2_type
        self.Mag_2_agency=Mag_2_agency
        self.Mag_3=Mag_3
        self.Mag_3_type=Mag_3_type
        self.Mag_3_agency=Mag_3_agency
    def __repr__(self):
        return "HEADER()"
    def __str__(self):
        print_str=str(self.time)+' '+str(self.latitude)+','+str(self.longitude)\
        +' '+str(self.depth)+' '+self.Mag_1_type+':'+str(self.Mag_1)+\
        self.Mag_2_type+':'+str(self.Mag_2)+\
        self.Mag_3_type+':'+str(self.Mag_3)+'  '+self.agency
        return print_str

def _int_conv(string):
    """
    Convenience tool to convert from string to integer, if empty string return
    a 999 rather than an error
    """
    try:
        intstring=int(string)
    except:
        intstring=999
    return intstring

def _float_conv(string):
    """
    Convenience tool to convert from string to float, if empty string return
    NaN rather than an error
    """
    try:
        floatstring=float(string)
    except:
        floatstring=float('NaN')
    return floatstring

def _str_conv(number, rounded=False):
    """
    Convenience tool to convert a number, either float or into into a string,
    if the int is 999, or the float is NaN, returns empty string.
    """
    if (type(number) == float and np.isnan(number)) or number == 999:
        string = ' '
    elif type(number) == str:
        return number
    elif not rounded:
        if number < 100000:
            string = str(number)
        else:
            exponant = int('{0:.2E}'.format(number).split('E+')[-1]) - 1
            divisor = 10 ** exponant
            string = '{0:.1f}'.format(number / divisor) + 'e' + str(exponant)
    elif rounded == 2:
        string = '{0:.2f}'.format(number)
    elif rounded == 1:
        string = '{0:.1f}'.format(number)
    return string

def readheader(sfilename):
    """
    Fucntion to read the header information from a seisan nordic format S-file.

    :type sfilename: str
    :param sfilename: Path to the s-file

    :returns: :class: EVENTINFO
    """
    import warnings
    f=open(sfilename,'r')
    # Base populate to allow for empty parts of file
    sfilename_header=EVENTINFO()
    topline=f.readline()
    if topline[79]==' ' or topline[79]=='1':
        # Topline contains event information
        try:
            sfile_seconds=int(topline[16:18])
            if sfile_seconds==60:
                sfile_seconds=0
                add_seconds=60
            else:
                add_seconds=0
            sfilename_header.time=UTCDateTime(int(topline[1:5]),int(topline[6:8]),
                                        int(topline[8:10]),int(topline[11:13]),
                                        int(topline[13:15]),sfile_seconds
                                        ,int(topline[19:20])*100000)+add_seconds
        except:
            warnings.warn("Couldn't read a date from sfile: "+sfilename)
            sfilename_header.time=UTCDateTime(0)
        sfilename_header.loc_mod_ind=topline[20]
        sfilename_header.dist_ind=topline[21]
        sfilename_header.ev_id=topline[22]
        sfilename_header.latitude=_float_conv(topline[23:30])
        sfilename_header.longitude=_float_conv(topline[31:38])
        sfilename_header.depth=_float_conv(topline[39:43])
        sfilename_header.depth_ind=topline[44]
        sfilename_header.loc_ind=topline[45]
        sfilename_header.agency=topline[46:48].strip()
        sfilename_header.nsta=_int_conv(topline[49:51])
        sfilename_header.t_RMS=_float_conv(topline[52:55])
        sfilename_header.Mag_1=_float_conv(topline[56:59])
        sfilename_header.Mag_1_type=topline[59]
        sfilename_header.Mag_1_agency=topline[60:63].strip()
        sfilename_header.Mag_2=_float_conv(topline[64:67])
        sfilename_header.Mag_2_type=topline[67]
        sfilename_header.Mag_2_agency=topline[68:71].strip()
        sfilename_header.Mag_3=_float_conv(topline[72:75])
        sfilename_header.Mag_3_type=topline[75].strip()
        sfilename_header.Mag_3_agency=topline[76:79].strip()
    else:
        for line in f:
            if line[79]=='1':
                line=topline
                try:
                    sfilename_header.time=UTCDateTime(int(topline[1:5]),int(topline[6:8]),
                                                int(topline[8:10]),int(topline[11:13]),
                                                int(topline[13:15]),int(topline[16:18])
                                                ,int(topline[19:20])*10)
                except:
                    sfilename_header.time=UTCDateTime(0)
                sfilename_header.loc_mod_ind=topline[21]
                sfilename_header.dist_ind=topline[22]
                sfilename_header.ev_id=topline[23]
                sfilename_header.latitude=_float_conv(topline[24:30])
                sfilename_header.longitude=_float_conv(topline[31:38])
                sfilename_header.depth=_float_conv(topline[39:43])
                sfilename_header.depth_ind=topline[44]
                sfilename_header.loc_ind=topline[45]
                sfilename_header.agency=topline[46:48].strip()
                sfilename_header.nsta=_int_conv(topline[49:51])
                sfilename_header.t_RMS=_float_conv(topline[52:55])
                sfilename_header.Mag_1=_float_conv(topline[56:59])
                sfilename_header.Mag_1_type=topline[60]
                sfilename_header.Mag_1_agency=topline[61:63].strip()
                sfilename_header.Mag_2=_float_conv(topline[64:67])
                sfilename_header.Mag_2_type=topline[68]
                sfilename_header.Mag_2_agency=topline[69:71].strip()
                sfilename_header.Mag_3=_float_conv(topline[72:75])
                sfilename_header.Mag_3_type=topline[76].strip()
                sfilename_header.Mag_3_agency=topline[77:79].strip()
            if line[79]=='7':
                break
    f.close()
    return sfilename_header

def readpicks(sfilename):
    """
    Function to read pick information from the s-file

    :type sfilename: String
    :param sfilename: Path to sfile

    :return: List of :class: PICK
    """
    # First we need to read the header to get the timing info
    sfilename_header=readheader(sfilename)
    evtime=sfilename_header.time
    f=open(sfilename,'r')
    pickline=[]
    lineno=0
    if 'headerend' in locals():
        del headerend
    for line in f:
        if 'headerend' in locals():
            if len(line.rstrip('\n').rstrip('\r')) in [80,79] and \
               (line[79]==' ' or line[79]=='4' or line [79]=='\n'):
                pickline+=[line]
        elif line[79]=='7':
            header=line
            headerend=lineno
        lineno+=1
    picks=[]
    for line in pickline:
        if line[18:28].strip()=='': # If line is empty miss it
            continue
        station=line[1:6].strip()
        channel=line[6:8].strip()
        impulsivity=line[9]
        weight=line[14]
        if weight=='_':
            phase=line[10:17]
            weight=''
            polarity=''
        else:
            phase=line[10:14].strip()
            polarity=line[16]
        try:
            time=UTCDateTime(evtime.year,evtime.month,evtime.day,
                             int(line[18:20]),int(line[20:22]),int(line[23:25]),
                             int(line[26:28])*10000)
        except (ValueError):
            time=UTCDateTime(evtime.year,evtime.month,evtime.day,
                             int(line[18:20]),int(line[20:22]),0,0)
            time+=60 # Add 60 seconds on to the time, this copes with s-file
        coda=_int_conv(line[28:33])
        amplitude=_float_conv(line[33:40])
        peri=_float_conv(line[41:45])
        azimuth=_float_conv(line[46:51])
        velocity=_float_conv(line[52:56])
        if header[57:60]=='AIN':
            SNR=''
            AIN=_float_conv(line[57:60])
        elif header[57:60]=='SNR':
            AIN=''
            SNR=_float_conv(line[57:60])
        azimuthres=_int_conv(line[60:63])
        timeres=_float_conv(line[63:68])
        finalweight=_int_conv(line[68:70])
        distance=_float_conv(line[70:75])
        CAZ=_int_conv(line[76:79])
        picks+=[PICK(station, channel, impulsivity, phase, weight, polarity,
                 time, coda, amplitude, peri, azimuth, velocity, AIN, SNR,
                 azimuthres, timeres, finalweight, distance, CAZ)]
    f.close()
    return picks

def readwavename(sfilename):
    """
    Convenience function to extract the waveform filename from the s-file,
    returns a list of waveform names found in the s-file as multiples can
    be present.

    :type sfilename: str
    :param sfilename: Path to the sfile

    :returns: List of str
    """
    f=open(sfilename)
    wavename=[]
    for line in f:
        if len(line)==81 and line[79]=='6':
            wavename.append(line[1:79].strip())
    f.close()
    return wavename

def blanksfile(wavefile,evtype,userID,outdir,overwrite=False, evtime=False):
    """
    Module to generate an empty s-file with a populated header for a given
    waveform.

    :type wavefile: String
    :param wavefile: Wavefile to associate with this S-file, the timing of the
                    S-file will be taken from this file if evtime is not set
    :type evtype: String
    :param evtype: L,R,D
    :type userID: String
    :param userID: 4-charectar SEISAN USER ID
    :type outdir: String
    :param outdir: Location to write S-file
    :type overwrite: Bool
    :param overwrite: Overwrite an existing S-file, default=False
    :type evtime: UTCDateTime
    :param evtime: If given this will set the timing of the S-file

    :returns: String, S-file name
    """

    from obspy import read as obsread
    import sys
    import os
    import datetime

    if not evtime:
        try:
            st=obsread(wavefile)
            evtime=st[0].stats.starttime
        except:
            print 'Wavefile: '+wavefile+' is invalid, try again with real data.'
            sys.exit()
    else:
        starttime=evtime
    # Check that user ID is the correct length
    if len(userID) != 4:
        print 'User ID must be 4 characters long'
        sys.exit()
    # Check that outdir exists
    if not os.path.isdir(outdir):
        print 'Out path does not exist, I will not create this: '+outdir
        sys.exit()
    # Check that evtype is one of L,R,D
    if evtype not in ['L','R','D']:
        print 'Event type must be either L, R or D'
        sys.exit()

    # Generate s-file name in the format dd-hhmm-ss[L,R,D].Syyyymm
    sfilename=outdir+'/'+str(evtime.day).zfill(2)+'-'+\
            str(evtime.hour).zfill(2)+\
            str(evtime.minute).zfill(2)+'-'+\
            str(evtime.second).zfill(2)+evtype+'.S'+\
            str(evtime.year)+\
            str(evtime.month).zfill(2)
    # Check is sfilename exists
    if os.path.isfile(sfilename) and not overwrite:
        print 'Desired sfilename: '+sfilename+' exists, will not overwrite'
        for i in range(1,20):
            sfilename=outdir+'/'+str(evtime.day).zfill(2)+'-'+\
                    str(evtime.hour).zfill(2)+\
                    str(evtime.minute).zfill(2)+'-'+\
                    str(evtime.second+i).zfill(2)+evtype+'.S'+\
                    str(evtime.year)+\
                    str(evtime.month).zfill(2)
            if not os.path.isfile(sfilename):
                break
        else:
            print 'Tried generated files up to 20s in advance and found they'
            print 'all exist, you need to clean your stuff up!'
            sys.exit()
        # sys.exit()
    f=open(sfilename,'w')
    # Write line 1 of s-file
    f.write(' '+str(evtime.year)+' '+\
            str(evtime.month).rjust(2)+\
            str(evtime.day).rjust(2)+' '+\
            str(evtime.hour).rjust(2)+\
            str(evtime.minute).rjust(2)+' '+\
            str(float(evtime.second)).rjust(4)+' '+\
            evtype+'1'.rjust(58)+'\n')
    # Write line 2 of s-file
    f.write(' ACTION:ARG '+str(datetime.datetime.now().year)[2:4]+'-'+\
            str(datetime.datetime.now().month).zfill(2)+'-'+\
            str(datetime.datetime.now().day).zfill(2)+' '+\
            str(datetime.datetime.now().hour).zfill(2)+':'+\
            str(datetime.datetime.now().minute).zfill(2)+' OP:'+\
            userID.ljust(4)+' STATUS:'+'ID:'.rjust(18)+\
            str(evtime.year)+\
            str(evtime.month).zfill(2)+\
            str(evtime.day).zfill(2)+\
            str(evtime.hour).zfill(2)+\
            str(evtime.minute).zfill(2)+\
            str(evtime.second).zfill(2)+\
            'I'.rjust(6)+'\n')
    # Write line 3 of s-file
    f.write(' '+wavefile+'6'.rjust(79-len(wavefile))+'\n')
    # Write final line of s-file
    f.write(' STAT SP IPHASW D HRMM SECON CODA AMPLIT PERI AZIMU'+\
            ' VELO AIN AR TRES W  DIS CAZ7\n')
    f.close()
    print 'Written s-file: '+sfilename
    return sfilename

def populateSfile(sfilename, picks):
    """
    Module to populate a blank nordic format S-file with pick information,
    arguments required are the filename of the blank s-file and the picks
    where picks is a dictionary of picks including station, channel,
    impulsivity, phase, weight, polarity, time, coda, amplitude, peri, azimuth,
    velocity, SNR, azimuth residual, Time-residual, final weight,
    epicentral distance & azimuth from event to station.

    This is a full pick line information from the seisan manual, P. 341

    :type sfilename: str
    :param sfilename: Path to S-file to populate, must have a header already
    :type picks: List of :class: PICK
    :param picks: List of the picks to be written out
    """

    f=open(sfilename, 'r')
    # Find type 7 line, under which picks should be - if there are already
    # picks there we should preserve them
    body=''
    header=''
    if 'headerend' in locals():
        del headerend
    for lineno, line in enumerate(f):
        identifier=line[79]
        if 'headerend' in locals():
            body+=line
        else:
            header+=line
        if identifier=='7':
            headerend=lineno
    f.close()
    #
    # Now generate lines for the new picks
    newpicks=''
    for pick in picks:
        if pick.distance >= 100.0:
            pick.distance=_int_conv(pick.distance)
        elif pick.distance < 100.0:
            pick.distance=int(round(pick.distance,1))
        newpicks+=pick.__str__()
        newpicks+='\n'
    # Write all new and old info back in
    f=open(sfilename, 'w')
    f.write(header)
    f.write(newpicks)
    f.write(body)
    f.write('\n'.rjust(81))
    f.close()
    return

def test_rw():
    """
    Function to test the functions herein.
    """
    import os
    test_pick=PICK('FOZ', 'SZ', 'I', 'P', '1', 'C', UTCDateTime("2012-03-26")+1,
                 coda=10, amplitude=0.2, peri=0.1,
                 azimuth=10.0, velocity=20.0, AIN=10, SNR='',
                 azimuthres=1, timeres=0.1,
                 finalweight=4, distance=10.0,
                 CAZ=2)
    print test_pick
    sfilename=blanksfile('test', 'L', 'TEST', '.', overwrite=True,\
     evtime=UTCDateTime("2012-03-26")+1)
    populateSfile(sfilename, [test_pick])
    assert readwavename(sfilename) == ['test']
    assert readpicks(sfilename)[0].station == test_pick.station
    assert readpicks(sfilename)[0].channel == test_pick.channel
    assert readpicks(sfilename)[0].impulsivity == test_pick.impulsivity
    assert readpicks(sfilename)[0].phase == test_pick.phase
    assert readpicks(sfilename)[0].weight == test_pick.weight
    assert readpicks(sfilename)[0].polarity== test_pick.polarity
    assert readpicks(sfilename)[0].time == test_pick.time
    assert readpicks(sfilename)[0].coda == test_pick.coda
    assert readpicks(sfilename)[0].amplitude == test_pick.amplitude
    assert readpicks(sfilename)[0].peri == test_pick.peri
    assert readpicks(sfilename)[0].azimuth == test_pick.azimuth
    assert readpicks(sfilename)[0].velocity == test_pick.velocity
    assert readpicks(sfilename)[0].AIN == test_pick.AIN
    assert readpicks(sfilename)[0].SNR == test_pick.SNR
    assert readpicks(sfilename)[0].azimuthres == test_pick.azimuthres
    assert readpicks(sfilename)[0].timeres == test_pick.timeres
    assert readpicks(sfilename)[0].finalweight == test_pick.finalweight
    assert readpicks(sfilename)[0].distance == test_pick.distance
    assert readpicks(sfilename)[0].CAZ == test_pick.CAZ
    header = readheader(sfilename)
    os.remove(sfilename)
    return True

if __name__=='__main__':
    # Read arguments
    import sys, os
    if len(sys.argv) != 6:
        print 'Requires 5 arguments: wavefile, evtype, userID, outdir, overwrite'
        sys.exit()
    else:
        wavefile=str(sys.argv[1])
        evtype=str(sys.argv[2])
        userID=str(sys.argv[3])
        outdir=str(sys.argv[4])
        overwrite=str(sys.argv[5])
    sfilename=blanksfile(wavefile,evtype,userID,outdir,overwrite)
    print sfilename
