import plotly.express as px

from plotly.offline import plot
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from hydroanalytics.precipitation.precipitation import *
from hydroanalytics.precipitation.tests.data import CVG_PRECIP

HERE = os.path.dirname(os.path.realpath(__file__))


class TestEvent(unittest.TestCase):
    def setUp(self) -> None:
        self.rainfall_data = pd.read_csv(
            CVG_PRECIP,
            index_col=0,
            parse_dates=True,
        )

        self.rainfall_data = self.rainfall_data[self.rainfall_data.index.year >= 2004]

    def test_get_events(self) -> None:
        events = get_events(self.rainfall_data, floor=0.001)
        events = events.loc[(events.precip_total > 0.1) & (events.duration > pd.Timedelta('1h'))]
        events_cp = get_event_attributes(
            latitude=39.0470,
            longitude=-84.6656,
            events=events,
        )

        self.events = events_cp

        self.events['duration_hours'] = self.events['duration'] / pd.Timedelta('1h')
        clustered_events = cluster_events(
            events=self.events,
            cluster_columns=['duration_hours', 'precip_total', 'precip_peak', 'return_period'],
            number_of_clusters=16,
        )

        clustered_events['cluster'] = clustered_events['cluster'].astype('category')

        fig = px.scatter_matrix(
            clustered_events,
            dimensions=['duration_hours', 'precip_total', 'precip_peak', 'return_period'],
            color="cluster",
            symbol="cluster",
            hover_data=['name', 'start', 'end', 'duration_hours', 'precip_total', 'precip_peak', 'return_period'],
        )

        plot(fig, filename='scatter_matrix.html')

        sub_plot_fig = make_subplots(
            rows=1,
            cols=1,
            shared_xaxes=True,
            x_title='Date and Time',
            row_titles=['', 'Precipitation (in)'],
        )

        # sub_plot_fig.add_trace(
        #     go.Bar(
        #         x=clustered_events['start'],
        #         y=np.full(shape=clustered_events.shape[0], fill_value=1),
        #         width=clustered_events['duration'],
        #     ),
        #     row=1,
        #     col=1,
        # )
        hour_resampled = self.rainfall_data.resample('1H').sum()
        sub_plot_fig.add_trace(
            go.Scatter(
                x=hour_resampled.index,
                y=hour_resampled['precip'],
                fill='tozeroy',
            ),
            row=1,
            col=1,
        )

        plot(figure_or_data=sub_plot_fig, filename='cluster_bar.html')
