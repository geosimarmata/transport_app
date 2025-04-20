import streamlit as st
import pandas as pd
import re
import difflib
from io import BytesIO

# Load historical trip data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('q_tp_record_2025-04-13T07_02_58.867855Z.csv')  # Make sure this file is in the same folder
        df['nama_shipper'] = df['nama_shipper'].fillna('Unknown')
        df = df[~df['trip_status'].isin(['Cancel', 'Unfulfill', 'Open'])]

        df['origin_norm'] = df['origin_location_name'].str.strip().str.lower()
        df['destination_norm'] = df['destination_location_name'].str.strip().str.lower()
        df['multi_drop_norm'] = df['multi_drop'].fillna('').str.lower()
        df['nama_shipper_norm'] = df['nama_shipper'].str.strip().str.lower()
        df['truck_type_norm'] = df['tipe_truk'].str.strip().str.lower()
        df['origin_city_norm'] = df['origin_city_name'].str.strip().str.lower()
        df['destination_city_norm'] = df['destination_city_name'].str.strip().str.lower()
        return df
    except FileNotFoundError:
        st.error("❌ Data file not found. Please make sure 'q_tp_record_2025-04-19.csv' is in the project directory.")
        return pd.DataFrame()  # Return empty DataFrame so the app doesn't break

data = load_data()


# --- SIDEBAR NAVIGATION ---
st.title("🚚 Vendor Tiering System")
st. sidebar.header("🔍 Navigation")
selected_section = st.sidebar.selectbox("Select Section", ['Tiering System', 'Vendor Filtering', 'Upload DO File'])

# ---------------- TIERING SYSTEM ----------------
if selected_section == 'Tiering System':
    shipper_options = ['ALL'] + sorted(data['nama_shipper'].dropna().unique().tolist())
    truck_options = ['ALL'] + sorted(data['tipe_truk'].dropna().unique().tolist())
    origin_location_options = ['ALL'] + sorted(data['origin_location_name'].dropna().unique().tolist())
    destination_location_options = ['ALL'] + sorted(data['destination_location_name'].dropna().unique().tolist())
    origin_city_options = ['ALL'] + sorted(data['origin_city_name'].dropna().unique().tolist())
    destination_city_options = ['ALL'] + sorted(data['destination_city_name'].dropna().unique().tolist())

    input_shippers = st.sidebar.multiselect("Shipper", shipper_options, default=['ALL'])
    input_truck_types = st.sidebar.multiselect("Truck Type", truck_options, default=['ALL'])
    input_origins = st.sidebar.multiselect("Origin Location", origin_location_options, default=['ALL'])
    input_destinations = st.sidebar.multiselect("Destination Location", destination_location_options, default=['ALL'])

    with st.sidebar.expander("⚙️ Advanced Filters (City-Level)", expanded=False):
        input_origin_cities = st.multiselect("Origin City", origin_city_options, default=['ALL'])
        input_destination_cities = st.multiselect("Destination City", destination_city_options, default=['ALL'])

    if st.button("▶️ Generate Tiering Results"):
        mask_origin = (data['origin_norm'].isin([o.strip().lower() for o in input_origins])) | ('ALL' in input_origins)
        mask_origin_city = (data['origin_city_norm'].isin([c.strip().lower() for c in input_origin_cities])) | ('ALL' in input_origin_cities)
        mask_truck_type = (data['truck_type_norm'].isin([t.strip().lower() for t in input_truck_types])) | ('ALL' in input_truck_types)
        mask_shipper = (data['nama_shipper_norm'].isin([s.strip().lower() for s in input_shippers])) | ('ALL' in input_shippers)

        if 'ALL' in input_destinations:
            mask_destination = pd.Series(True, index=data.index)
        else:
            dest_list = [d.strip().lower() for d in input_destinations]
            pattern = '|'.join(map(re.escape, dest_list))
            dest_norm_cond = data['destination_norm'].isin(dest_list)
            multi_drop_cond = data['multi_drop_norm'].str.contains(pattern, na=False)
            mask_destination = dest_norm_cond | multi_drop_cond

        mask_destination_city = (data['destination_city_norm'].isin([c.strip().lower() for c in input_destination_cities])) | ('ALL' in input_destination_cities)

        matched_routes = data[ 
            mask_origin & mask_origin_city & mask_truck_type & 
            mask_shipper & mask_destination & mask_destination_city
        ]

        if not matched_routes.empty:
            if 'ALL' in input_destinations:
                matched_routes['matched_destination_display'] = matched_routes['destination_location_name']
            else:
                matched_routes['matched_destination_display'] = matched_routes.apply(
                    lambda row: ', '.join([d for d in input_destinations if d.strip().lower() in row['multi_drop_norm']])
                    if any(d.strip().lower() in row['multi_drop_norm'] for d in input_destinations)
                    else row['destination_location_name'],
                    axis=1
                )

            vendor_stats = matched_routes.groupby('nama_vendor').agg({
                'tanggal_muat': ['count', 'max']
            }).reset_index()
            vendor_stats.columns = ['nama_vendor', 'trip_count', 'last_used_date']
            vendor_stats = vendor_stats.sort_values(by=['trip_count', 'last_used_date'], ascending=[False, False])

            st.subheader("🏆 Top Vendors for the Selected Route:")
            st.write(vendor_stats.head(10))

            st.subheader("📋 Detailed Delivery Records:")
            detailed_records = matched_routes[[
                'nama_vendor', 'origin_location_name', 'matched_destination_display',
                'origin_city_name', 'destination_city_name', 'tipe_truk', 'tanggal_muat', 'multi_drop', 'nama_shipper'
            ]].rename(columns={'matched_destination_display': 'destination_location_display'})
            st.write(detailed_records)
        else:
            st.warning("No delivery records matched the selected filters. Try relaxing the conditions.")

# ---------------- VENDOR FILTERING ----------------
elif selected_section == 'Vendor Filtering':
    st.sidebar.subheader("🔍 Filter by Vendor")
    vendor_list = ['ALL'] + sorted(data['nama_vendor'].dropna().unique().tolist())
    selected_vendor = st.sidebar.selectbox("Select Vendor", vendor_list)

    if st.button("▶️ Show Vendor Summary"):
        if selected_vendor != 'ALL':
            st.subheader(f"🚚 Routes for {selected_vendor}")
            vendor_routes = data[data['nama_vendor'] == selected_vendor]

            vendor_route_summary = vendor_routes.groupby(
                ['origin_location_name', 'destination_location_name', 'tipe_truk']
            ).size().reset_index(name='trip_count')

            st.write(vendor_route_summary.sort_values(by='trip_count', ascending=False))
            st.write(f"Total trips made: {vendor_routes.shape[0]}")
            st.write(f"Truck types used: {', '.join(vendor_routes['tipe_truk'].unique())}")
        else:
            st.info("Select a vendor from the sidebar to see details.")

# ---------------- UPLOAD DO FILE SECTION ----------------
elif selected_section == 'Upload DO File':
    st.subheader("📄 Upload Delivery Order (DO) File")
    uploaded_file = st.file_uploader("Choose an Excel file", type=["xls", "xlsx"])

    if uploaded_file:
        do_data = pd.read_excel(uploaded_file)
        st.write("📋 Uploaded DO Preview:")
        st.write(do_data.head())

        # Prevent duplicate processing
        if 'DO_processed_key' not in st.session_state:
            st.session_state.DO_processed_key = set()

        def fuzzy_match(val, options, cutoff=0.6):
            match = difflib.get_close_matches(val.strip().lower(), options, n=1, cutoff=cutoff)
            return match[0] if match else None

        def recommend_vendors_tiered(do_row, historical_data, max_per_tier=5):
            origin = do_row['origin_location_name']
            destination = do_row['destination_location_name']
            truck = do_row['tipe_truk']
            shipper = do_row['nama_shipper']

            matches = historical_data[
                (historical_data['origin_location_name'] == origin) &
                (historical_data['destination_location_name'] == destination) &
                (historical_data['tipe_truk'] == truck) &
                (historical_data['nama_shipper'] == shipper)
            ]

            # Fallback: fuzzy match origin/destination
            if matches.empty:
                origin_match = fuzzy_match(origin, historical_data['origin_location_name'].dropna().unique())
                dest_match = fuzzy_match(destination, historical_data['destination_location_name'].dropna().unique())
                if origin_match and dest_match:
                    matches = historical_data[
                        (historical_data['origin_location_name'] == origin_match) &
                        (historical_data['destination_location_name'] == dest_match) &
                        (historical_data['tipe_truk'] == truck)
                    ]

            # Fallback: city-level
            if matches.empty:
                matches = historical_data[
                    (historical_data['origin_city_name'] == do_row['origin_city_name']) &
                    (historical_data['destination_city_name'] == do_row['destination_city_name']) &
                    (historical_data['tipe_truk'] == truck)
                ]

            if matches.empty:
                return "", "", ""

            vendor_counts = matches['nama_vendor'].value_counts()
            tier1 = vendor_counts.head(max_per_tier).index.tolist()
            tier2 = vendor_counts.iloc[max_per_tier:max_per_tier*2].index.tolist()
            tier3 = vendor_counts.iloc[max_per_tier*2:max_per_tier*3].index.tolist()

            return ", ".join(tier1), ", ".join(tier2), ", ".join(tier3)

        if st.button("▶️ Generate Vendor Recommendation"):
            new_rows = do_data[
                ~do_data['trip_id'].astype(str).isin(st.session_state.DO_processed_key)
            ]
            st.session_state.DO_processed_key.update(new_rows['trip_id'].astype(str))

            new_rows[['Tier 1 Vendors', 'Tier 2 Vendors', 'Tier 3 Vendors']] = new_rows.apply(
                lambda row: pd.Series(recommend_vendors_tiered(row, data)), axis=1
            )

            st.success("Vendor recommendations generated for new DO entries.")
            st.write(new_rows)

            # Store for download
            to_download = new_rows.copy()
            to_download_file = BytesIO()
            to_download.to_excel(to_download_file, index=False)
            to_download_file.seek(0)

            st.download_button(
                label="💾 Download Recommended DO File",
                data=to_download_file,
                file_name="recommended_do.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
