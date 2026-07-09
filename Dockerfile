FROM mcr.microsoft.com/dotnet/aspnet:10.0 AS base
WORKDIR /app
EXPOSE 8080

FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src
COPY ["AzureInferenceApi/AzureInferenceApi.csproj", "AzureInferenceApi/"]
COPY ["StockExchangeEA.Schemas/StockExchangeEA.Schemas.csproj", "StockExchangeEA.Schemas/"]
RUN dotnet restore "AzureInferenceApi/AzureInferenceApi.csproj" -c Release -o /app/build

COPY . .
WORKDIR "/src/AzureInferenceApi"
RUN dotnet build "AzureInferenceApi.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "AzureInferenceApi.csproj" -c Release -o /app/publish

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "AzureInferenceApi.dll"]