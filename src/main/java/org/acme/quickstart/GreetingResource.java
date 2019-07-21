package org.acme.quickstart;

import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;

import javax.inject.Inject;
import javax.ws.rs.Consumes;
import javax.ws.rs.GET;
import javax.ws.rs.HeaderParam;
import javax.ws.rs.POST;
import javax.ws.rs.Path;
import javax.ws.rs.PathParam;
import javax.ws.rs.Produces;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.MultivaluedMap;
import javax.ws.rs.core.Response;

import org.jboss.resteasy.plugins.providers.multipart.InputPart;
import org.jboss.resteasy.plugins.providers.multipart.MultipartFormDataInput;

@Path("/hello")
public class GreetingResource {

    @Inject
    GreetingService service;

    @GET
    @Produces(MediaType.TEXT_PLAIN)
    public String hello() {
        return "hello";
    }

    @GET
    @Produces(MediaType.TEXT_PLAIN)
    @Path("/greeting/{name}")
    public String greeting(@PathParam("name") String name) {
        return service.greeting(name);
    }
    @POST
    @Path("/labelImage")
    @Consumes("multipart/form-data")
    public Response loadImage(@HeaderParam("Content-Length") String contentLength, MultipartFormDataInput input) throws Exception {
        long before = System.currentTimeMillis();
        InputPart inputPart = input.getFormDataMap().get("file").iterator().next();
        String fileName = parseFileName(inputPart.getHeaders());
        byte[] bytes = streamToByte(inputPart.getBody(InputStream.class, null), Integer.parseInt(contentLength));
        String returnString = (new LabelImage()).labelImage(fileName, bytes);
        return Response.status(200).entity("labeled image = " + returnString + "time=" + (System.currentTimeMillis() - before)).build();
    }

    @POST
    @Path("/labelImageStatic")
    @Consumes("multipart/form-data")
    public Response loadImageStatic(@HeaderParam("Content-Length") String contentLength, MultipartFormDataInput input) throws Exception {
        long before = System.currentTimeMillis();
        InputPart inputPart = input.getFormDataMap().get("file").iterator().next();
        String fileName = parseFileName(inputPart.getHeaders());
        byte[] bytes = streamToByte(inputPart.getBody(InputStream.class, null), Integer.parseInt(contentLength));
        String returnString = (new LabelImageStatic()).labelImage(fileName, bytes);
        return Response.status(200).entity("labeled image (static)= " + returnString + "time=" + (System.currentTimeMillis() - before)).build();
    }

    // Parse Content-Disposition header to get the original file name
    private static String parseFileName(MultivaluedMap<String, String> headers) {
        String[] contentDispositionHeader = headers.getFirst("Content-Disposition").split(";");
        for (String name : contentDispositionHeader) {
            if ((name.trim().startsWith("filename"))) {
                String[] tmp = name.split("=");
                String fileName = tmp[1].trim().replaceAll("\"", "");
                return fileName;
            }
        }
        return "randomName";
    }

    
    private static final int MAX_BUFFER_SIZE = Integer.MAX_VALUE - 8;
    private static final int BUFFER_SIZE = 8192;
    private static byte[] streamToByte(InputStream source, int contentLength) throws IOException {
    	int capacity = contentLength; //TODO maybe a more clever algorithm for determining buffer initial size?
        byte[] buf = new byte[capacity];
        int nread = 0;
        int n;
        for (;;) {
            // read to EOF which may read more or less than initialSize (eg: file
            // is truncated while we are reading)
            while ((n = source.read(buf, nread, capacity - nread)) > 0)
                nread += n;

            // if last call to source.read() returned -1, we are done
            // otherwise, try to read one more byte; if that failed we're done too
            if (n < 0 || (n = source.read()) < 0)
                break;

            // one more byte was read; need to allocate a larger buffer
            if (capacity <= MAX_BUFFER_SIZE - capacity) {
                capacity = Math.max(capacity << 1, BUFFER_SIZE);
            } else {
                if (capacity == MAX_BUFFER_SIZE)
                    throw new OutOfMemoryError("Required array size too large");
                capacity = MAX_BUFFER_SIZE;
            }
            buf = Arrays.copyOf(buf, capacity);
            buf[nread++] = (byte)n;
        }
        return (capacity == nread) ? buf : Arrays.copyOf(buf, nread);
    }
}