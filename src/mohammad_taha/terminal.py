"""Plain-text presentation shared by interactive and scripted CLI runs."""


def heading(title):
    print(f"\n{title}\n{'=' * len(title)}")


def context_report(context):
    heading("Weather and market prices")
    weather = context["weather"]
    prices = context["prices"]
    print(f"  Weather  {weather['status'].title()}")
    data = weather["data"]
    if data:
        print(f"           {data['temperature_c']:.1f} °C  |  {data['humidity_pct']:.0f}% humidity")
        print(f"           Observed UTC  {data['observed_at_utc']}")

    print(f"  Prices   {prices['status'].title()}")
    data = prices["data"]
    if data:
        print(f"           {data['commodity']} in {data['market']}")
        print(f"           {data['price']:,.2f} {data['currency']} per {data['unit']}")
        print(f"           Observation date  {data['observation_date']}")
        if data.get("previous_price") is not None:
            print(f"           Previous quote   {data['previous_price']:,.2f} {data['currency']}")


def inspection_report(folder, summary, context):
    heading("Inspection results")
    for line in summary.splitlines():
        print(f"  {line}")
    context_report(context)
    heading("Exported reports")
    print(f"  Report folder   {folder}")
    print(f"  Item results    {folder / 'items.csv'}")
    print(f"  Figures         {folder / 'figures'}")
