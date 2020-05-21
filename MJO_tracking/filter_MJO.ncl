load "$NCARG_ROOT/lib/ncarg/nclscripts/contrib/kf_filter.ncl"

begin

ifile = "trmm_daily_anomalies_1998-2019.nc"

f1 = addfile(ifile,"r")

var = f1->precipitation(:,:,{-15:15})

lat = var&lat
lon = var&lon
time = var&time


mis = -999
mis@_FillValue = -999

obsPerDay = 1
tMin = 20
tMax = 100

kMin = 0
kMax = 9

hMin = mis
hMax = mis

wavename = "none"
printVarSummary(var)

;if( calcMjo ) then
;   print_clock( "Filtering MJO..." )

;   mjo = data
;   mjo@long_name     = "Madden-Julian Oscillation in " + str_upper(varName)
;   if( filtName.eq."wk99" ) then
;    mjo@filter        = "Wheeler & Kiladis (1999)"
;     mjo@wavenumber    = (/   1,   5 /)
;     mjo@period        = (/  30,  96 /)
;     mjo@depth         = (/ mis, mis /)
;   else
;     mjo@filter        = "Kiladis et al. (2005 JAS) for 20-100"
;     mjo@wavenumber    = (/   0,   9 /)
;     mjo@period        = (/  20, 100 /)
;     mjo@depth         = (/ mis, mis /)
;   end if

;  do y = 0, ( dimsizes(lat) - 1 )
;    print_clock( y + " " + ( dimsizes(lat) - 1 ) + " " )

;   mjo(time|:,{lat|lat(y)},lon|:) = (/ kf_filter(  \\
;     data(time|:,{lat|lat(y)},lon|:), obsPerDay,  \\
;     mjo@period(0),     mjo@period(1),     \\
;      mjo@wavenumber(0), mjo@wavenumber(1), \\
;      mjo@depth(0),      mjo@depth(1),      \\
;     "none" ) /)

;   end do

;    print_clock( "Writing MJO... " )
;    fout->mjo = mjo

print_clock( "Filtering MJO..." )

mjo = var
mjo@long_name     = "Madden-Julian Oscillation in OLR" 

do i = 0, (dimsizes(lat)-1)
   mjo(time|:,lon|:,{lat|lat(i)}) = kf_filter(var(time|:,lon|:,{lat|lat(i)}),obsPerDay,tMin,tMax,kMin,kMax,hMin,hMax,wavename)
end do

mjo_time_lon = dim_avg_n_Wrap(mjo,(/2/))

system("rm -f MJO_filtered.nc")
fo = addfile("MJO_filtered.nc", "c")

fo->mjo_time_lon=mjo_time_lon

end


;cdo -sellonlatbox,-40,-120,-90,90 out.nc out1.nc
;ncwa -a lat out1.nc MJO_filtered.nc
