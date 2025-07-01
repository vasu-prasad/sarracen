from pandas import Series
from astropy import units as u

from .sarracen_dataframe import SarracenDataFrame


class SarracenSeries(Series):
    _metadata = ['_units']

    def __init__(self, data=None, unit=u.dimensionless_unscaled,
                 *args, **kwargs):

        if isinstance(data, u.Quantity):
            unit = data.unit
            data = data.value
        elif not isinstance(unit, u.UnitBase):
            raise TypeError("unit must be an astropy UnitBase instance.")

        super().__init__(data, *args, **kwargs)

        self.__unit = unit

    def __repr__(self):
        base_repr = super().__repr__()
        unit_repr = f", unit: {self.__unit}"
        return base_repr + unit_repr

    @property
    def _constructor(self):
        return SarracenSeries

    @property
    def _constructor_expanddim(self):
        return SarracenDataFrame

    @property
    def unit(self):
        return self.__unit

    # unit conversions

    def to(self, new_unit):
        if not isinstance(new_unit, u.UnitBase):
            raise TypeError("Unit must be an astropy UnitBase instance.")
        converted_values = (self.values * self.unit).to(new_unit).value
        return SarracenSeries(converted_values, unit=new_unit,
                              index=self.index)

    def si(self):
        si_unit = (1 * self.unit).si.unit
        return self.to(si_unit)

    def cgs(self):
        cgs_unit = (1 * self.unit).cgs.unit
        return self.to(cgs_unit)

    # override arithmetic operations

    def __add__(self, other):
        if (isinstance(other, SarracenSeries) and
                self.unit.is_equivalent(other.unit)):
            new_values = self.values + other.to(self.unit).values
            return SarracenSeries(new_values, unit=self.unit, index=self.index)
        else:
            raise u.UnitConversionError(
                "Can only add SarracenSeries with compatible units.")

    def __sub__(self, other):
        if (isinstance(other, SarracenSeries) and
                self.unit.is_equivalent(other.unit)):
            new_values = self.values - other.to(self.unit).values
            return SarracenSeries(new_values, unit=self.unit, index=self.index)
        else:
            raise u.UnitConversionError(
                "Can only subtract SarracenSeries with compatible units.")

    def __mul__(self, other):
        if isinstance(other, SarracenSeries):
            new_unit = self.unit * other.unit
            return SarracenSeries(
                self.values * other.values,
                unit=new_unit,
                index=self.index
            )
        elif isinstance(other, u.Quantity):
            new_data = (self.values * self.unit * other).to(self.unit *
                                                            other.unit)
            return SarracenSeries(new_data.value, unit=new_data.unit,
                                  index=self.index)
        elif isinstance(other, (u.UnitBase, u.CompositeUnit)):
            return SarracenSeries(self.values, unit=self.unit * other,
                                  index=self.index)
        else:
            return SarracenSeries(super().__mul__(other), unit=self.unit,
                                  index=self.index)

    def __truediv__(self, other):
        if isinstance(other, SarracenSeries):
            new_unit = self.unit / other.unit
            return SarracenSeries(self.values / other.values, unit=new_unit,
                                  index=self.index)
        elif isinstance(other, u.Quantity):
            new_data = (
                self.values * self.unit / other
            ).to(self.unit / other.unit)
            return SarracenSeries(new_data.value, unit=new_data.unit,
                                  index=self.index)
        elif isinstance(other, (u.UnitBase, u.CompositeUnit)):
            return SarracenSeries(self.values, unit=self.unit / other,
                                  index=self.index)
        else:
            return SarracenSeries(super().__truediv__(other), unit=self.unit,
                                  index=self.index)

    def __pow__(self, other):
        if isinstance(other, (int, float)):
            new_unit = self.unit ** other
            return SarracenSeries(self.values ** other, unit=new_unit,
                                  index=self.index)
        elif isinstance(other, (SarracenSeries, u.Quantity)):
            if other.unit.is_equivalent(u.dimensionless_unscaled):
                new_unit = self.unit ** other.values
                return SarracenSeries(self.values ** other.values,
                                      unit=new_unit, index=self.index)
            else:
                raise u.UnitTypeError(
                    "Can only raise something to a dimensionless quantity"
                )
        else:
            return SarracenSeries(
                super().__pow__(other),
                unit=u.dimensionless_unscaled,
                index=self.index
            )
